"""Public-facing :func:`convert_model` implementation.

The function exposed at ``teradata_opus_translate.convert_model`` lives
here. The thin re-export at the package root pulls it from this module.

API surface (locked for v1.0.0)
-------------------------------

``convert_model`` accepts a HuggingFace Marian model id OR a local path
to a downloaded HF repo, and writes a single self-contained ONNX file
with an embedded ``com.microsoft.BeamSearch`` op.

The function is **deliberately narrow**: parameters tunable at SQL time
via BYOM ``Const_*`` USING parameters (``num_beams``, ``max_length``,
``min_length``, ``length_penalty``, ``repetition_penalty``,
``num_return_sequences``) are NOT part of the export-time API. They
remain as graph inputs in the produced ONNX so the SQL layer can
override them per query. See ``docs/decisions.md`` ADR for the
rationale.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from teradata_opus_translate._source import is_local_source, resolve_source

LOGGER = logging.getLogger(__name__)

# Default verification samples per source-language family. Picked so the
# parity check covers a few typical sentence shapes without depending on
# the caller supplying anything. Keep these short; the verification step
# loads transformers and runs ``MarianMTModel.generate()`` per sample.
#
# Coverage rationale: the keys span (a) the source-language space of the
# curated ``Helsinki-NLP/opustranslate`` HuggingFace collection, plus (b)
# every ISO 639-1 source code that appears in five or more
# ``Helsinki-NLP/opus-mt-*`` repos in the wider Helsinki-NLP catalogue
# (see ``helsinki-nlp-models.csv`` for the catalogue snapshot the
# selection was derived from). Both 2-letter (ISO 639-1) and 3-letter
# (ISO 639-3) keys are present because Helsinki-NLP mixes the two: the
# curated tiny models in ``opustranslate`` use 3-letter codes
# (``opus-mt_tiny_deu-eng``) while the bulk of ``opus-mt-{src}-{tgt}``
# repos use 2-letter codes (``opus-mt-de-en``). This keeps Phase 3 bulk
# conversion from silently falling back to generic English samples for
# real production language pairs. The generic fallback is reserved for
# rare/unknown source codes and local-path-with-no-parseable-id
# scenarios -- see ``_GENERIC_FALLBACK_SAMPLES`` below.
#
# Each language gets exactly 3 short, varied samples (greeting / weather
# / "I am reading a book about machine learning") in the native script
# with proper diacritics. Marian / SentencePiece handles non-Latin
# scripts natively; transliteration would defeat the parity check.
_DEFAULT_SAMPLES_BY_SRC_LANG: dict[str, list[str]] = {
    # --- The original Phase 1 set (kept verbatim). ---
    "de": [
        "Hallo Welt.",
        "Das Wetter ist heute schön.",
        "Ich lese ein Buch über Maschinelles Lernen.",
    ],
    "fr": [
        "Bonjour le monde.",
        "Il fait beau aujourd'hui.",
        "Je lis un livre sur l'apprentissage automatique.",
    ],
    "es": [
        "Hola mundo.",
        "Hoy hace buen tiempo.",
        "Estoy leyendo un libro sobre aprendizaje automático.",
    ],
    "it": [
        "Ciao mondo.",
        "Oggi il tempo è bello.",
        "Sto leggendo un libro sull'apprendimento automatico.",
    ],
    "en": [
        "Hello, world.",
        "The weather is nice today.",
        "I am reading a book about machine learning.",
    ],
    # --- 3-letter aliases for the curated opustranslate collection. ---
    "deu": [
        "Hallo Welt.",
        "Das Wetter ist heute schön.",
        "Ich lese ein Buch über Maschinelles Lernen.",
    ],
    "fra": [
        "Bonjour le monde.",
        "Il fait beau aujourd'hui.",
        "Je lis un livre sur l'apprentissage automatique.",
    ],
    "spa": [
        "Hola mundo.",
        "Hoy hace buen tiempo.",
        "Estoy leyendo un libro sobre aprendizaje automático.",
    ],
    "ita": [
        "Ciao mondo.",
        "Oggi il tempo è bello.",
        "Sto leggendo un libro sull'apprendimento automatico.",
    ],
    "eng": [
        "Hello, world.",
        "The weather is nice today.",
        "I am reading a book about machine learning.",
    ],
    # --- Western & Northern Europe (Latin script). ---
    "nl": [
        "Hallo wereld.",
        "Het weer is vandaag mooi.",
        "Ik lees een boek over machinaal leren.",
    ],
    "nld": [
        "Hallo wereld.",
        "Het weer is vandaag mooi.",
        "Ik lees een boek over machinaal leren.",
    ],
    "pt": [
        "Olá, mundo.",
        "O tempo está bom hoje.",
        "Estou lendo um livro sobre aprendizado de máquina.",
    ],
    "ca": [
        "Hola món.",
        "Avui fa bon temps.",
        "Estic llegint un llibre sobre aprenentatge automàtic.",
    ],
    "cat": [
        "Hola món.",
        "Avui fa bon temps.",
        "Estic llegint un llibre sobre aprenentatge automàtic.",
    ],
    "gl": [
        "Ola mundo.",
        "Hoxe fai bo tempo.",
        "Estou a ler un libro sobre aprendizaxe automática.",
    ],
    "glg": [
        "Ola mundo.",
        "Hoxe fai bo tempo.",
        "Estou a ler un libro sobre aprendizaxe automática.",
    ],
    "eu": [
        "Kaixo mundua.",
        "Gaur eguraldi ona dago.",
        "Ikasketa automatikoari buruzko liburu bat irakurtzen ari naiz.",
    ],
    "eus": [
        "Kaixo mundua.",
        "Gaur eguraldi ona dago.",
        "Ikasketa automatikoari buruzko liburu bat irakurtzen ari naiz.",
    ],
    "ro": [
        "Salut lume.",
        "Vremea este frumoasă astăzi.",
        "Citesc o carte despre învățarea automată.",
    ],
    "sv": [
        "Hej världen.",
        "Vädret är fint idag.",
        "Jag läser en bok om maskininlärning.",
    ],
    "da": [
        "Hej verden.",
        "Vejret er godt i dag.",
        "Jeg læser en bog om maskinlæring.",
    ],
    "no": [
        "Hei verden.",
        "Været er fint i dag.",
        "Jeg leser en bok om maskinlæring.",
    ],
    "is": [
        "Halló heimur.",
        "Veðrið er gott í dag.",
        "Ég er að lesa bók um vélanám.",
    ],
    "fi": [
        "Hei maailma.",
        "Sää on tänään hyvä.",
        "Luen kirjaa koneoppimisesta.",
    ],
    "et": [
        "Tere maailm.",
        "Ilm on täna ilus.",
        "Loen raamatut masinõppest.",
    ],
    "lt": [
        "Labas pasauli.",
        "Šiandien geras oras.",
        "Skaitau knygą apie mašininį mokymąsi.",
    ],
    "lv": [
        "Sveika pasaule.",
        "Šodien ir labs laiks.",
        "Es lasu grāmatu par mašīnmācīšanos.",
    ],
    "pl": [
        "Witaj świecie.",
        "Dziś jest ładna pogoda.",
        "Czytam książkę o uczeniu maszynowym.",
    ],
    "cs": [
        "Ahoj světe.",
        "Dnes je hezké počasí.",
        "Čtu knihu o strojovém učení.",
    ],
    "sk": [
        "Ahoj svet.",
        "Dnes je pekné počasie.",
        "Čítam knihu o strojovom učení.",
    ],
    "sl": [
        "Pozdravljen, svet.",
        "Vreme je danes lepo.",
        "Berem knjigo o strojnem učenju.",
    ],
    "hu": [
        "Helló világ.",
        "Ma szép idő van.",
        "Egy könyvet olvasok a gépi tanulásról.",
    ],
    "mt": [
        "Hello dinja.",
        "It-temp huwa sabiħ illum.",
        "Qed naqra ktieb dwar it-tagħlim awtomatiku.",
    ],
    "sq": [
        "Përshëndetje botë.",
        "Sot bën mot i mirë.",
        "Po lexoj një libër për mësimin e makinerisë.",
    ],
    "af": [
        "Hallo wêreld.",
        "Die weer is vandag mooi.",
        "Ek lees 'n boek oor masjienleer.",
    ],
    "eo": [
        "Saluton mondo.",
        "La vetero estas bela hodiaŭ.",
        "Mi legas libron pri maŝinlernado.",
    ],
    # --- Slavic / Cyrillic. ---
    "ru": [
        "Привет, мир.",
        "Сегодня хорошая погода.",
        "Я читаю книгу о машинном обучении.",
    ],
    "rus": [
        "Привет, мир.",
        "Сегодня хорошая погода.",
        "Я читаю книгу о машинном обучении.",
    ],
    "uk": [
        "Привіт, світе.",
        "Сьогодні гарна погода.",
        "Я читаю книгу про машинне навчання.",
    ],
    "bg": [
        "Здравей, свят.",
        "Времето днес е хубаво.",
        "Чета книга за машинното обучение.",
    ],
    "mk": [
        "Здраво свету.",
        "Денес времето е убаво.",
        "Читам книга за машинско учење.",
    ],
    # --- Greek. ---
    "el": [
        "Γεια σου κόσμε.",
        "Ο καιρός είναι ωραίος σήμερα.",
        "Διαβάζω ένα βιβλίο για τη μηχανική μάθηση.",
    ],
    "ell": [
        "Γεια σου κόσμε.",
        "Ο καιρός είναι ωραίος σήμερα.",
        "Διαβάζω ένα βιβλίο για τη μηχανική μάθηση.",
    ],
    # --- Semitic / RTL. ---
    "ar": [
        "مرحبا بالعالم.",
        "الطقس جميل اليوم.",
        "أنا أقرأ كتابا عن التعلم الآلي.",
    ],
    "ara": [
        "مرحبا بالعالم.",
        "الطقس جميل اليوم.",
        "أنا أقرأ كتابا عن التعلم الآلي.",
    ],
    "he": [
        "שלום עולם.",
        "מזג האוויר נעים היום.",
        "אני קורא ספר על למידת מכונה.",
    ],
    # --- Turkic / Caucasus. ---
    "tr": [
        "Merhaba dünya.",
        "Bugün hava güzel.",
        "Makine öğrenmesi hakkında bir kitap okuyorum.",
    ],
    "tur": [
        "Merhaba dünya.",
        "Bugün hava güzel.",
        "Makine öğrenmesi hakkında bir kitap okuyorum.",
    ],
    # --- East Asian. ---
    "ja": [
        "こんにちは、世界。",
        "今日は天気がいいです。",
        "機械学習についての本を読んでいます。",
    ],
    "ko": [
        "안녕하세요, 세계.",
        "오늘 날씨가 좋습니다.",
        "저는 기계 학습에 관한 책을 읽고 있습니다.",
    ],
    "kor": [
        "안녕하세요, 세계.",
        "오늘 날씨가 좋습니다.",
        "저는 기계 학습에 관한 책을 읽고 있습니다.",
    ],
    "zh": [
        "你好,世界。",
        "今天天气很好。",
        "我正在读一本关于机器学习的书。",
    ],
    "zho": [
        "你好,世界。",
        "今天天气很好。",
        "我正在读一本关于机器学习的书。",
    ],
    "vi": [
        "Xin chào thế giới.",
        "Hôm nay thời tiết đẹp.",
        "Tôi đang đọc một cuốn sách về học máy.",
    ],
    "id": [
        "Halo dunia.",
        "Cuaca hari ini bagus.",
        "Saya sedang membaca buku tentang pembelajaran mesin.",
    ],
    # --- Caribbean / Creoles. ---
    "ht": [
        "Bonjou mond.",
        "Tan an bèl jodi a.",
        "M ap li yon liv sou aprantisaj otomatik.",
    ],
    # --- Sub-Saharan Africa. ---
    "ha": [
        "Sannu duniya.",
        "Yanayi yana da kyau yau.",
        "Ina karanta littafi game da koyon na'ura.",
    ],
    "yo": [
        "Bawo aye.",
        "Oju ojo dara loni.",
        "Mo n ka iwe nipa ẹkọ ẹrọ.",
    ],
    "ig": [
        "Ndewo ụwa.",
        "Ihu igwe dị mma taa.",
        "Ana m agụ akwụkwọ banyere mmụta igwe.",
    ],
    "sn": [
        "Mhoro nyika.",
        "Mamiriro ekunze akanaka nhasi.",
        "Ndiri kuverenga bhuku rinotaura nezvekudzidza kwemuchina.",
    ],
    "rw": [
        "Muraho isi.",
        "Ikirere ni cyiza uyu munsi.",
        "Ndimo nsoma igitabo cyerekeye kwiga kw'imashini.",
    ],
    "rn": [
        "Muraho isi.",
        "Ikirere ni ciza uyu musi.",
        "Ndi gusoma igitabu cerekeye kwiga kw'imashini.",
    ],
    "lg": [
        "Mwasuze mutya nsi.",
        "Embeera y'obudde nnungi leero.",
        "Nsoma ekitabo ekikwata ku kuyiga kwa kyuma.",
    ],
    "sg": [
        "Bara ala dunia.",
        "Bê tî ngu ayeke nzoni laso.",
        "Mbi yeke diko mbeti so ayeke ndo ti mandango ti masini.",
    ],
    "st": [
        "Lumela lefatše.",
        "Boemo ba leholimo bo botle kajeno.",
        "Ke bala buka e mabapi le ho ithuta ha mochini.",
    ],
    "ts": [
        "Avuxeni misava.",
        "Maxelo ya tikweni ma sasekile namuntlha.",
        "Ndzi hlaya buku hi vudyondzi bya muchini.",
    ],
    # --- Southeast Asian / Pacific (Latin script). ---
    "tl": [
        "Kamusta mundo.",
        "Maganda ang panahon ngayon.",
        "Nagbabasa ako ng libro tungkol sa machine learning.",
    ],
    "to": [
        "Mālō e lelei māmani.",
        "ʻOku lelei e ʻea he ʻahó ni.",
        "ʻOku ou lau ha tohi fekauʻaki mo e ako fakaʻotomētiki.",
    ],
    "ee": [
        "Ndo xexeame.",
        "Yame nyo egbe.",
        "Mele agbalẽ aɖe xlẽm tso mɔ̃kpɔkplɔ̃ ŋu.",
    ],
}

# Reserved for the rare/unknown case: a source code that isn't in the
# default-samples table above. Reasons this can happen at runtime:
#
# - The caller pointed ``source`` at a local directory whose path does
#   not encode a parseable language code (``/data/my-marian-dump/``).
# - The model id uses a non-standard prefix not handled by
#   :func:`_infer_source_lang` (e.g. ``opus-mt-tc-big-...`` with an
#   unusual src code, or fully custom forks).
# - A new ISO code outside the Phase 3 catalogue snapshot ships.
#
# In all of these the safe move is to verify with English-ish input and
# warn the caller (via ``LOGGER``) that they probably want to pass
# ``verify_samples=`` explicitly. The default-samples table above covers
# every common path, so this fallback is no longer the hot case.
_GENERIC_FALLBACK_SAMPLES = [
    "Hello, world.",
    "The quick brown fox jumps over the lazy dog.",
]


@dataclass(frozen=True)
class ParityResult:
    """Token-parity comparison against ``MarianMTModel.generate()``.

    A mismatch count of zero means every sample produced an identical
    token-id sequence under both paths (after canonicalising the trailing
    EOS, which BeamSearch omits and HF emits -- see
    ``teradata-byom-onnx-seq2seq`` skill).
    """

    samples: list[str]
    hf_token_ids: list[list[int]]
    onnx_token_ids: list[list[int]]
    mismatches: int = field(default=0)


@dataclass(frozen=True)
class ConvertModelResult:
    """Summary returned by :func:`convert_model`.

    Attributes
    ----------
    output_path:
        Resolved absolute path of the written ``.onnx`` file.
    size_bytes:
        Size of the written ONNX file, in bytes.
    source:
        The original ``source`` argument (HF id or path) as resolved
        for ``from_pretrained``.
    source_kind:
        ``"local"`` if ``source`` was a local directory, otherwise
        ``"hf"``.
    precision:
        Precision mode used for the export (``"fp32"`` or ``"int8"``).
    parity:
        Token-parity result against ``MarianMTModel.generate()``. ``None``
        if ``verify=False``. When present, contains per-sample token-id
        sequences from both paths plus a ``mismatches`` count -- 0 means
        all samples matched.
    """

    output_path: Path
    size_bytes: int
    source: str
    source_kind: Literal["hf", "local"]
    precision: Literal["fp32", "int8"]
    parity: ParityResult | None = None


def _default_samples(source_lang: str | None) -> list[str]:
    """Pick verification samples for the given source language code.

    The Helsinki-NLP id format is ``Helsinki-NLP/opus-mt-{src}-{tgt}``
    (or ``opus-mt_tiny_{src3}-{tgt3}`` for the curated
    ``opustranslate`` collection); :func:`_infer_source_lang` returns
    the ``src`` portion. ``_DEFAULT_SAMPLES_BY_SRC_LANG`` covers every
    source language in that collection plus every ISO 639-1 source code
    appearing in 5+ ``opus-mt-*`` repos in the wider Helsinki-NLP
    catalogue, so the common Phase 3 paths land on language-appropriate
    samples.

    The ``_GENERIC_FALLBACK_SAMPLES`` path is reserved for the rare
    case where the source language cannot be inferred (local path
    without a parseable id, fully custom fork, or a code outside the
    catalogue snapshot). Callers landing in that branch should pass
    ``verify_samples=`` explicitly for meaningful coverage.
    """
    if source_lang and source_lang in _DEFAULT_SAMPLES_BY_SRC_LANG:
        return list(_DEFAULT_SAMPLES_BY_SRC_LANG[source_lang])
    return list(_GENERIC_FALLBACK_SAMPLES)


def _infer_source_lang(source: str) -> str | None:
    """Best-effort source-language extraction from an HF model id.

    Recognised patterns (in order):

    * ``Helsinki-NLP/opus-mt-{src}-{tgt}`` -> ``src`` (2-letter ISO
      639-1 in the bulk of the catalogue, e.g. ``de`` from
      ``opus-mt-de-en``).
    * ``Helsinki-NLP/opus-mt_tiny_{src3}-{tgt3}`` -> ``src3`` (3-letter
      ISO 639-3 used by the curated ``opustranslate`` collection, e.g.
      ``deu`` from ``opus-mt_tiny_deu-eng``).
    * ``Helsinki-NLP/opus-mt-tc-{src}-{tgt}`` and
      ``Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}`` (Tatoeba-Challenge
      transformer models) -> ``src``.

    Returns ``None`` for anything else (local paths, fully custom
    forks). The default-samples table keys both 2-letter and 3-letter
    variants for every supported language, so callers landing on either
    pattern get language-appropriate samples.
    """
    name = source.rsplit("/", 1)[-1]
    # opus-mt_tiny_{src3}-{tgt3} (curated opustranslate collection).
    if name.startswith("opus-mt_tiny_"):
        rest = name[len("opus-mt_tiny_") :]
        parts = rest.split("-")
        if len(parts) < 2:
            return None
        return parts[0]
    if not name.startswith("opus-mt-"):
        return None
    rest = name[len("opus-mt-") :]
    # Strip the Tatoeba-Challenge "tc"/"tc-big" prefix so we land on the
    # real source-language code rather than treating "tc" as the source.
    if rest.startswith("tc-big-"):
        rest = rest[len("tc-big-") :]
    elif rest.startswith("tc-"):
        rest = rest[len("tc-") :]
    parts = rest.split("-")
    if len(parts) < 2:
        return None
    return parts[0]


# 2-letter ISO 639-1 -> 3-letter ISO 639-3 map for the source/target codes
# that appear in published ``Helsinki-NLP/opus-mt-*`` repos.  Phase 3's
# calibration loader (see ``calibration.py``) only knows 3-letter codes
# because Tatoeba_mt configs use the 3-letter form.  This map covers the
# 25 published ``opus-mt_tiny_*`` pairs (which are already 3-letter, so the
# map is a no-op for them) and the legacy ``opus-mt-de-en``-style repos
# used by the slow-lane int8 test fixture.  Extend the map (and document
# the reason) when a new pair needs static calibration.
_ISO1_TO_ISO3 = {
    "ar": "ara",
    "ca": "cat",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "eu": "eus",
    "fr": "fra",
    "gl": "glg",
    "it": "ita",
    "ko": "kor",
    "nl": "nld",
    "ru": "rus",
    "tr": "tur",
    "zh": "zho",
}


def _normalise_lang_code(code: str | None) -> str | None:
    """Return a 3-letter ISO 639-3 code for ``code`` if known, else ``None``.

    Pass-through for already-3-letter codes; lookup via :data:`_ISO1_TO_ISO3`
    for 2-letter codes.
    """
    if code is None:
        return None
    if len(code) == 3 and code.lower() == code:
        return code
    return _ISO1_TO_ISO3.get(code.lower())


def _infer_pair_3letter(source: str) -> tuple[str, str] | None:
    """Infer ``(src3, tgt3)`` 3-letter language pair from an HF model id.

    Returns ``None`` if the pair cannot be derived (custom forks, local
    paths without a Helsinki-NLP naming convention, etc.).  Recognised
    patterns mirror :func:`_infer_source_lang` but include the target code:

    * ``Helsinki-NLP/opus-mt_tiny_{src3}-{tgt3}`` (curated collection)
    * ``Helsinki-NLP/opus-mt-{src1}-{tgt1}`` (bulk Helsinki-NLP catalogue)
    * ``Helsinki-NLP/opus-mt-tc[-big]-{src}-{tgt}`` (Tatoeba-Challenge)
    """
    name = source.rsplit("/", 1)[-1]
    if name.startswith("opus-mt_tiny_"):
        rest = name[len("opus-mt_tiny_") :]
        parts = rest.split("-")
        if len(parts) < 2:
            return None
        src3 = _normalise_lang_code(parts[0])
        tgt3 = _normalise_lang_code(parts[1])
        if src3 and tgt3:
            return src3, tgt3
        return None
    if not name.startswith("opus-mt-"):
        return None
    rest = name[len("opus-mt-") :]
    if rest.startswith("tc-big-"):
        rest = rest[len("tc-big-") :]
    elif rest.startswith("tc-"):
        rest = rest[len("tc-") :]
    parts = rest.split("-")
    if len(parts) < 2:
        return None
    src3 = _normalise_lang_code(parts[0])
    tgt3 = _normalise_lang_code(parts[1])
    if src3 and tgt3:
        return src3, tgt3
    return None


def _setup_logging(verbose: bool, log_level: str | int | None) -> None:
    """Lightweight logging configuration for CLI / one-shot callers."""
    if log_level is not None:
        level = log_level
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    # Only configure the package logger so we don't stomp on caller config.
    pkg_logger = logging.getLogger("teradata_opus_translate")
    pkg_logger.setLevel(level)
    # Avoid duplicate handlers on repeat calls in the same process.
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)


def convert_model(
    source: str | os.PathLike[str],
    *,
    precision: Literal["fp32", "int8"] = "fp32",
    output_path: str | os.PathLike[str],
    opset: int = 14,
    ir_version: int = 8,
    verify: bool = True,
    verify_samples: list[str] | None = None,
    no_repeat_ngram_size: int | None = None,
    early_stopping: bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
    log_level: str | int | None = None,
    calibration_pair: str | None = None,
) -> ConvertModelResult:
    """Convert a HuggingFace Marian model to a single ONNX file with
    embedded ``com.microsoft.BeamSearch``.

    Parameters
    ----------
    source:
        HuggingFace repo id (e.g. ``"Helsinki-NLP/opus-mt-de-en"``) OR a
        path to a local directory containing a downloaded HF repo.
        Detection is automatic: an existing directory is treated as a
        local source; everything else is passed to ``from_pretrained``
        as an HF id.
    precision:
        ``"fp32"`` (default) or ``"int8"``.  The ``"int8"`` path applies
        the weight-only int8 rewriter (v1.1.0, Issue #160) to the
        encoder and decoder subgraphs *before* they are composed into
        the ``com.microsoft.BeamSearch`` wrapper graph (the BeamSearch
        op is opaque to ORT's quantizer, so we cannot quantize the
        assembled file in one shot).  Every ``MatMul`` whose B input is
        a 2-D fp32 const initializer is rewritten to store the weight
        as int8 + a per-channel symmetric scale, with a
        ``DequantizeLinear`` node inserted at inference time.
        Activations stay fp32; no calibration corpus is consulted.
        Output is roughly half the fp32 size (~90 MiB vs ~170 MiB on
        the curated ``opus-mt_tiny_*`` collection; the embedding
        tables stay fp32 and the BeamSearch wrapper is small but
        non-zero overhead) with a small token-parity divergence on a
        minority of samples; see ``docs/decisions.md`` for the full
        recipe and tolerance.

        The earlier dynamic / static activation-quantization recipes
        were removed in v1.1.0 -- they collapsed beam search on the
        ``*-eng`` Marian pairs under BYOM 7.0.0.4's pinned ORT 1.13.1.
        See PR #138, Issue #140, and the v1.1.0 CHANGELOG entry.
    output_path:
        Destination ``.onnx`` file path. Parent directories are created
        if missing; the file is overwritten if it already exists.
    opset:
        ONNX opset version for the encoder / decoder subgraphs. Defaults
        to ``14``, the BYOM ORT 1.16.3 ceiling. Increase only if you've
        verified BYOM compatibility for the new version.
    ir_version:
        ONNX IR version stamped on the produced top-level graph. Default
        ``8`` -- matches BYOM 7.x's bundled ORT (1.16.3 lineage). BYOM
        7.0.4 rejects IR >= 9 with
        ``Unsupported model IR version: 9, max supported IR version: 8``,
        so the wheel pins this to 8 even though the upstream
        ``torch.onnx.export`` default is now 9. Set to ``9`` (or higher)
        only if your downstream ORT supports newer IRs. See
        ``docs/decisions.md`` Decision 12.
    verify:
        If ``True`` (default), runs full token-parity verification
        against ``MarianMTModel.generate()`` on each sample in
        ``verify_samples`` (or a sensible default set if not supplied).
        Always runs when true regardless of model size.
    verify_samples:
        Source-language strings to use during verification. If ``None``,
        the package picks a default set per source-language family
        (German, French, Spanish, Italian, English) inferred from the
        HF id. Pass an explicit list when ``source`` is a local path or
        when the model is a non-standard pair.
    no_repeat_ngram_size:
        Baked into the BeamSearch graph **as a node attribute** at
        export time. Defaults to ``model.config.no_repeat_ngram_size``.
        Cannot be overridden at SQL-scoring time -- BYOM does not expose
        this knob via ``Const_*``.
    early_stopping:
        Baked into the BeamSearch graph **as a node attribute** at
        export time. Defaults to ``model.config.early_stopping``.
        Cannot be overridden at SQL-scoring time.
    cache_dir:
        Optional HuggingFace cache directory passed through to
        ``from_pretrained``.
    verbose:
        If ``True``, configure the package logger at INFO level.
    log_level:
        Explicit logging level; takes precedence over ``verbose``.
    calibration_pair:
        Deprecated as of v1.1.0 (Issue #160).  The v1.0.x int8 static
        quantization recipe took this as a Tatoeba corpus selector; the
        v1.1.0 weight-only int8 recipe does not consult a calibration
        corpus, so the kwarg is now a no-op.  Passing a non-None value
        triggers a ``DeprecationWarning``.  Retained on the signature
        for backward compatibility with v1.0.x callers; will be removed
        in a future major release.

    Returns
    -------
    ConvertModelResult
        Dataclass with ``output_path``, ``size_bytes``, resolved
        ``source``, ``source_kind`` (``"hf"`` / ``"local"``),
        ``precision``, and (if ``verify=True``) a populated ``parity``
        result.

    Raises
    ------
    RuntimeError
        If the produced ONNX would exceed 2 GiB. v1 does not support
        ``external_data``; ``precision="int8"`` is the recommended
        escape hatch (roughly half the fp32 size on opus-mt-de-en;
        actual ratio depends on the model's MatMul:embedding weight
        balance).
    AssertionError
        If ``verify=True`` and any sample produces a divergent token-id
        sequence between ``MarianMTModel.generate()`` and the exported
        ONNX. The error message lists the first divergent sample and
        the position of the first differing token. **The fp32 path
        requires byte-identical token IDs.** The int8 path applies a
        looser tolerance to allow for quantization-induced drift; see
        the int8 entry in ``docs/decisions.md`` for the concrete
        threshold.

    Notes
    -----
    Parameters tunable at SQL time via BYOM ``Const_*`` USING parameters
    (``num_beams``, ``max_length``, ``min_length``, ``length_penalty``,
    ``repetition_penalty``, ``num_return_sequences``) are deliberately
    NOT part of this API. They remain as graph inputs in the produced
    ONNX so the SQL layer can override them per query. The local
    parity-verification step uses sensible defaults
    (``num_beams=4``, ``max_length=256``, ``min_length=1``,
    ``length_penalty=1.0``, ``repetition_penalty=1.0``,
    ``num_return_sequences=1``).
    """
    if precision not in ("fp32", "int8"):
        raise ValueError(f"Unsupported precision={precision!r}; expected 'fp32' or 'int8'.")

    _setup_logging(verbose, log_level)

    # Heavy imports kept inside the call so a bare ``import`` of the
    # package stays cheap.
    from transformers import MarianMTModel

    from teradata_opus_translate import __version__ as _pkg_version
    from teradata_opus_translate._converter.assemble import assemble_full_model

    resolved = resolve_source(source)
    src_kind: Literal["hf", "local"] = "local" if is_local_source(source) else "hf"

    LOGGER.info("Loading Marian model: %s (kind=%s)", resolved, src_kind)
    from_pretrained_kwargs: dict[str, object] = {}
    if cache_dir is not None:
        from_pretrained_kwargs["cache_dir"] = os.fspath(cache_dir)
    model = MarianMTModel.from_pretrained(resolved, **from_pretrained_kwargs).eval()

    # v1.1.0 (Issue #160): the int8 recipe is weight-only and does not
    # consult a calibration corpus.  ``calibration_pair`` is retained on
    # the public signature for backward compatibility -- the v1.0.x int8
    # static-quantization recipe took it as a corpus selector.  Warn
    # callers that pass a non-None value so they can drop it from their
    # invocation; it is now a no-op.
    if calibration_pair is not None:
        import warnings

        warnings.warn(
            "calibration_pair is deprecated and ignored as of v1.1.0; the int8 "
            "recipe is now weight-only and does not consult a calibration corpus. "
            "Remove the kwarg from your convert_model() call (Issue #160).",
            DeprecationWarning,
            stacklevel=2,
        )

    cfg = model.config
    nrns = (
        no_repeat_ngram_size
        if no_repeat_ngram_size is not None
        else int(getattr(cfg, "no_repeat_ngram_size", 0) or 0)
    )
    early = (
        early_stopping if early_stopping is not None else bool(getattr(cfg, "early_stopping", True))
    )

    out = Path(os.fspath(output_path)).expanduser().resolve()
    LOGGER.info(
        "Assembling ONNX (precision=%s opset=%d ir_version=%d "
        "no_repeat_ngram_size=%d early_stopping=%s) -> %s",
        precision,
        opset,
        ir_version,
        nrns,
        early,
        out,
    )
    written = assemble_full_model(
        model,
        out,
        opset=opset,
        no_repeat_ngram_size=nrns,
        early_stopping=early,
        package_version=_pkg_version,
        precision=precision,
        ir_version=ir_version,
    )
    size = written.stat().st_size
    LOGGER.info("Wrote %s (%d bytes / %.2f MiB)", written, size, size / (1024 * 1024))

    parity_result: ParityResult | None = None
    if verify:
        # Pick samples up front so we can include them in the result
        # regardless of pass / fail.
        samples = verify_samples
        if samples is None:
            samples = _default_samples(_infer_source_lang(resolved))
        LOGGER.info("Running token-parity verification on %d sample(s)", len(samples))
        parity_result = _verify_token_parity(
            model=model,
            resolved_source=resolved,
            onnx_path=written,
            samples=samples,
        )
        _check_parity_tolerance(parity_result, precision=precision, samples=samples)

    return ConvertModelResult(
        output_path=written,
        size_bytes=size,
        source=resolved,
        source_kind=src_kind,
        precision=precision,
        parity=parity_result,
    )


# ---------------------------------------------------------------------------
# Parity tolerance per precision
# ---------------------------------------------------------------------------
#
# fp32: byte-identical token IDs are required. The export is mathematically
# faithful, so any divergence is a converter bug (cf. Decision 9 in
# ``docs/decisions.md``).
#
# int8: weight-only quantization (v1.1.0, Issue #160) is lossy because each
# rewritten ``MatMul`` reconstructs an int8-then-dequantized weight at
# inference time.  The per-channel symmetric scale recovers the original
# fp32 weight to within ``max(|W[:,c]|) / 254`` per channel, which is
# small enough that decoded text is almost always byte-identical to fp32
# -- but a minority of samples drift at the token-id level on small /
# tricky-input pairs.  The locked v1.1.0 tolerance (see the int8 decision
# entry in ``docs/decisions.md``) is:
#
#   * For *every* sample, the first ``_INT8_PREFIX_MUST_MATCH`` token IDs
#     must match exactly. The first slot is always the BOS / decoder
#     start token (58100 for Marian); the second slot is the first
#     content token, whose match is a strong signal that the encoder
#     output, embedding lookup, and seed-step decoder all survived
#     the weight-only rewrite.
#   * The number of samples whose full token-id sequence diverges must
#     be at most ``max(1, ceil(N * _INT8_MAX_MISMATCH_FRACTION))`` where
#     N is the sample count. The ``max(1, ...)`` floor handles small
#     sample sets (the default ``de`` set ships 3 samples; 10% of 3 is
#     0.3 which would round down to 0 and refuse the empirically-known
#     1-sample drift). For the 3-sample German default we allow 1
#     mismatch; for a 10-sample set we allow 1; for a 20-sample set we
#     allow 2.
#
# Empirical evidence for the 10% threshold (Phase 5 Gate 3 report on
# branch ``160-phase5-weight-only-int8``, 100-sentence sweep per pair
# across the 25 curated ``opus-mt_tiny_*`` pairs):
#
#   * 20 / 25 pairs PASS: byte-identical 92-100 / 100, BLEU mean 96.6+.
#   * 3 / 25 NEEDS-REVIEW: byte-identical 92-95 / 100, no broken decodes.
#   * 2 / 25 BROKEN (deu-eng, ell-eng): trigram-runaway patterns on a
#     minority of inputs (24-31 / 100); shipped anyway because the bulk
#     of decodes are clean and these are known-limited failure modes.
#
# The 10% bound is tight enough to flag a regression on a PASS pair (the
# typical sweep produces 0-3 mismatches per 100) while loose enough to
# absorb the 1/3 mismatch the 3-sample default verification routinely
# lands on for the more challenging pairs.  Tightening below 10% would
# refuse the empirical baseline; loosening above 10% would let a
# regression slip through.  The Gate 1 anchor test (eng-nld Dubai
# donation) is byte-identical fp32==int8, so the eng-nld default
# verification passes with 0 mismatches.
#
# Both thresholds are pinned in the dedicated int8 test
# (``tests/test_int8_quantization.py``); changes need a corresponding
# decision-log update.

_INT8_PREFIX_MUST_MATCH = 2
_INT8_MAX_MISMATCH_FRACTION = 0.10


def _check_parity_tolerance(
    parity: ParityResult,
    *,
    precision: Literal["fp32", "int8"],
    samples: list[str],
) -> None:
    """Apply the per-precision parity tolerance to a ``ParityResult``.

    For ``precision="fp32"`` we require *zero* mismatches -- this is the
    Decision 9 contract and matches the historical Phase 1 guarantee.

    For ``precision="int8"`` we apply a two-part tolerance:

    1. Every sample's first ``_INT8_PREFIX_MUST_MATCH`` token IDs must
       match exactly (so quantization-induced drift cannot land on the
       very first token, which would suggest the embedding lookup or the
       seed decoder step is broken rather than just slightly noisy).
    2. The fraction of samples that diverge end-to-end must be at most
       ``_INT8_MAX_MISMATCH_FRACTION``.

    Raises :class:`AssertionError` with a precise diagnostic if either
    threshold is exceeded. The message names the first offending sample
    and the position of the first divergent token so failures are
    actionable.
    """
    import math

    if precision == "fp32":
        max_total_mismatches = 0
        prefix_must_match = max(
            (len(ids) for ids in parity.hf_token_ids),
            default=0,
        )
    elif precision == "int8":
        # v1.1.0 weight-only recipe.  ``max(1, ceil(...))`` so the
        # small-N case (3-sample German default) still permits the
        # empirically-observed 1-sample drift; see the tolerance
        # comment block below for the gate-evidence rationale.
        max_total_mismatches = max(1, math.ceil(len(samples) * _INT8_MAX_MISMATCH_FRACTION))
        prefix_must_match = _INT8_PREFIX_MUST_MATCH
    else:  # pragma: no cover - guarded earlier
        raise ValueError(f"Unsupported precision={precision!r}")

    # --- Hard prefix check: must hold for *every* sample. ---
    for i, (sample, hf_ids, onnx_ids) in enumerate(
        zip(parity.samples, parity.hf_token_ids, parity.onnx_token_ids, strict=True)
    ):
        prefix = min(prefix_must_match, len(hf_ids), len(onnx_ids))
        for j in range(prefix):
            if hf_ids[j] != onnx_ids[j]:
                raise AssertionError(
                    f"Token-parity prefix check failed (precision={precision}): "
                    f"sample index {i} diverged at position {j} "
                    f"(prefix tolerance: first {prefix} tokens must match).\n"
                    f"Sample: {sample!r}\n"
                    f"  HF   ids ({len(hf_ids)}):   {hf_ids}\n"
                    f"  ONNX ids ({len(onnx_ids)}): {onnx_ids}"
                )

    # --- Total-mismatch budget. ---
    if parity.mismatches > max_total_mismatches:
        # Find the first end-to-end divergent sample for the diagnostic.
        for i, (sample, hf_ids, onnx_ids) in enumerate(
            zip(parity.samples, parity.hf_token_ids, parity.onnx_token_ids, strict=True)
        ):
            if hf_ids != onnx_ids:
                div = next(
                    (j for j, (a, b) in enumerate(zip(hf_ids, onnx_ids, strict=False)) if a != b),
                    min(len(hf_ids), len(onnx_ids)),
                )
                raise AssertionError(
                    "Token-parity verification failed: "
                    f"{parity.mismatches}/{len(samples)} sample(s) diverged "
                    f"(precision={precision}, tolerance={max_total_mismatches}).\n"
                    f"First divergent sample (index {i}): {sample!r}\n"
                    f"  HF   ids ({len(hf_ids)}):   {hf_ids}\n"
                    f"  ONNX ids ({len(onnx_ids)}): {onnx_ids}\n"
                    f"  first divergence at position {div}"
                )


# Verification-time defaults for the SQL-tunable parameters.  These are
# passed both as ONNX runtime inputs and as ``generate()`` kwargs so the
# two paths run with identical configuration.  These values are NOT
# baked into the exported model -- they are the recommended defaults
# users should set via BYOM ``Const_*`` USING params at scoring time.
_VERIFY_NUM_BEAMS = 4
_VERIFY_MAX_LENGTH = 256
_VERIFY_MIN_LENGTH = 1
_VERIFY_NUM_RETURN_SEQUENCES = 1
_VERIFY_LENGTH_PENALTY = 1.0
_VERIFY_REPETITION_PENALTY = 1.0


def _verify_token_parity(
    *,
    model: object,
    resolved_source: str,
    onnx_path: Path,
    samples: list[str],
) -> ParityResult:
    """Run each sample through both ``MarianMTModel.generate()`` and the
    exported ONNX BeamSearch graph; report token-id matches.

    The trailing EOS / pad-to-max convention differences (HF emits a
    trailing EOS, BeamSearch omits it and pads with pad_token_id) are
    canonicalised before comparison so the assertion isolates real
    divergences.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(resolved_source)
    cfg = model.config  # type: ignore[attr-defined]
    pad_token_id = int(cfg.pad_token_id)
    eos_token_id = int(cfg.eos_token_id)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # The v1.1.0 weight-only int8 recipe keeps ``num_beams`` as a top-level
    # graph input (BYOM ``Const_num_beams`` USING-clause tunable) for both
    # fp32 and int8 artifacts.  The Phase 4 (#153) int8 build path used to
    # bake ``num_beams=4`` into the BeamSearch contrib op; that path was
    # removed in v1.1.0 with the rest of the static-quant recipe.  We
    # still inspect the session signature so the verifier can run against
    # v1.0.x int8 artifacts a caller might point us at.
    onnx_input_names = {i.name for i in sess.get_inputs()}
    num_beams_is_baked = "num_beams" not in onnx_input_names

    hf_all: list[list[int]] = []
    onnx_all: list[list[int]] = []
    mismatches = 0

    for sample in samples:
        # --- HuggingFace path ---
        enc = tokenizer(sample, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(  # type: ignore[attr-defined]
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=_VERIFY_NUM_BEAMS,
                min_length=_VERIFY_MIN_LENGTH,
                max_length=_VERIFY_MAX_LENGTH,
                num_return_sequences=_VERIFY_NUM_RETURN_SEQUENCES,
                length_penalty=_VERIFY_LENGTH_PENALTY,
                repetition_penalty=_VERIFY_REPETITION_PENALTY,
                no_repeat_ngram_size=int(getattr(cfg, "no_repeat_ngram_size", 0) or 0),
                early_stopping=bool(getattr(cfg, "early_stopping", True)),
                # The BeamSearch contrib op does not implement these.
                bad_words_ids=None,
                forced_eos_token_id=None,
                renormalize_logits=False,
                do_sample=False,
            )
        hf_ids = _strip_padding(out[0].tolist(), pad_token_id, eos_token_id)
        hf_all.append(hf_ids)

        # --- ONNX path ---
        # ``num_return_sequences`` is intentionally absent from the feeds:
        # as of v1.0.1 it is baked into the graph as a ``Constant(1)``
        # node and is no longer a top-level input. Passing it here would
        # raise ``InvalidArgument: Invalid input name``. See Issue #82.
        np_enc = tokenizer(sample, return_tensors="np")
        feeds = {
            "input_ids": np_enc["input_ids"].astype(np.int32),
            "attention_mask": np_enc["attention_mask"].astype(np.int32),
            "min_length": np.array([_VERIFY_MIN_LENGTH], dtype=np.int32),
            "max_length": np.array([_VERIFY_MAX_LENGTH], dtype=np.int32),
            "length_penalty": np.array([_VERIFY_LENGTH_PENALTY], dtype=np.float32),
            "repetition_penalty": np.array([_VERIFY_REPETITION_PENALTY], dtype=np.float32),
        }
        if not num_beams_is_baked:
            feeds["num_beams"] = np.array([_VERIFY_NUM_BEAMS], dtype=np.int32)
        sequences = sess.run(None, feeds)[0]  # (1, num_return_sequences, max_length)
        onnx_ids = _strip_padding(sequences[0, 0].tolist(), pad_token_id, eos_token_id)
        onnx_all.append(onnx_ids)

        if hf_ids != onnx_ids:
            mismatches += 1

    return ParityResult(
        samples=list(samples),
        hf_token_ids=hf_all,
        onnx_token_ids=onnx_all,
        mismatches=mismatches,
    )


def _strip_padding(
    token_ids: list[int],
    pad_token_id: int,
    eos_token_id: int,
) -> list[int]:
    """Strip trailing pad tokens AND a single trailing EOS, if present.

    Mirrors the canonicalisation in ``tests/test_converter_smoke.py``:
    ``MarianMTModel.generate()`` emits EOS then pads; the BeamSearch
    contrib op omits EOS and pads. Both decode to the same string under
    ``skip_special_tokens=True``; canonicalising before token-id diff
    isolates real divergences from the marker-emission convention.
    """
    out = list(token_ids)
    while out and out[-1] == pad_token_id:
        out.pop()
    if out and out[-1] == eos_token_id:
        out.pop()
    return out
