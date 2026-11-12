from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_PATH = Path("AudioWM_Project_Status_Summary.docx")


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_page_margins(section, top=0.9, bottom=0.9, left=0.95, right=0.95):
    section.top_margin = Inches(top)
    section.bottom_margin = Inches(bottom)
    section.left_margin = Inches(left)
    section.right_margin = Inches(right)


def style_base(document):
    styles = document.styles

    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)

    title = styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0x1F, 0x3C, 0x5A)

    subtitle = styles["Subtitle"]
    subtitle.font.name = "Aptos"
    subtitle.font.size = Pt(11)
    subtitle.font.italic = True
    subtitle.font.color.rgb = RGBColor(0x5A, 0x66, 0x73)

    for name, size in [("Heading 1", 15), ("Heading 2", 11.5)]:
        style = styles[name]
        style.font.name = "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1F, 0x3C, 0x5A)


def add_body_paragraph(document, text, italic=False):
    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(7)
    run = p.add_run(text)
    run.italic = italic
    return p


def add_bullets(document, items):
    for item in items:
        p = document.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.add_run(item)


def add_small_heading(document, text):
    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(0x1F, 0x3C, 0x5A)
    return p


def style_table(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)


def add_methods_table(document):
    table = document.add_table(rows=1, cols=2)
    style_table(table)
    table.columns[0].width = Inches(2.1)
    table.columns[1].width = Inches(4.9)

    hdr = table.rows[0].cells
    hdr[0].text = "Component"
    hdr[1].text = "Final Implementation"
    for cell in hdr:
        set_cell_shading(cell, "DCEAF7")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True

    rows = [
        ("Analysis / synthesis", "20 ms frames, 10 ms hop, Hann window, 1024-point FFT, overlap-add reconstruction."),
        ("Embedding band", "Mid-band speech region from 600 Hz to 2500 Hz to balance perceptual masking and channel survival."),
        ("Payload", "8 payload bits, each repeated over 4 consecutive symbol frames for robustness."),
        ("Watermark family", "Spread-spectrum style magnitude perturbation using a deterministic pseudorandom spreading code regenerated from a shared seed."),
        ("Synchronization", "13-bit Barker-code preamble with normalized correlation detection and one-step lattice confirmation."),
        ("Protocol realism", "Shared protocol dictionary defines packet structure, key seed, framing, and detector constants on both encoder and decoder."),
        ("Channel model", "Telephone-like distortions including 300-3400 Hz bandlimit, 7.5 kHz resampling, 4-bit quantization, and additive noise toggles."),
    ]

    for left, right in rows:
        cells = table.add_row().cells
        cells[0].text = left
        cells[1].text = right


def add_results_table(document):
    table = document.add_table(rows=1, cols=4)
    style_table(table)
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(1.25)
    table.columns[2].width = Inches(1.15)
    table.columns[3].width = Inches(2.1)

    hdr = table.rows[0].cells
    hdr[0].text = "Case"
    hdr[1].text = "Decoded Bits"
    hdr[2].text = "BER"
    hdr[3].text = "Notes"
    for cell in hdr:
        set_cell_shading(cell, "DCEAF7")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True

    rows = [
        ("MomAndBald.wav, clean path", "11001100", "0.0", "Known good demonstration case under the current protocol and sync logic."),
        ("Librispeech_03_speaker_2803_6.1s.wav", "11001100", "0.0", "Recovered correctly in clean evaluation after nonzero embedding offset."),
        ("Librispeech_10_speaker_84_8.2s.wav", "11001100", "0.0", "Another clean recovery case showing the method can generalize beyond one speaker clip."),
        ("Librispeech_04_speaker_6295_6.8s.wav", "10101100", "0.125", "Near-lock failure: payload was almost recovered, but synchronization fell back late in the stream."),
    ]

    for case, bits, ber, notes in rows:
        cells = table.add_row().cells
        cells[0].text = case
        cells[1].text = bits
        cells[2].text = ber
        cells[3].text = notes


def add_failure_table(document):
    table = document.add_table(rows=1, cols=3)
    style_table(table)
    table.columns[0].width = Inches(2.0)
    table.columns[1].width = Inches(2.1)
    table.columns[2].width = Inches(2.8)

    hdr = table.rows[0].cells
    hdr[0].text = "Failure Mode"
    hdr[1].text = "Observed Symptom"
    hdr[2].text = "Interpretation"
    for cell in hdr:
        set_cell_shading(cell, "F3E4CC")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True

    rows = [
        ("False packet candidates", "Decoder often produced roughly twice as many packet candidates as actual transmitted packets.", "Single-peak preamble detection is still too willing to trigger on speech-dependent structure."),
        ("Weak validation", "Wrong payloads could occasionally satisfy the ad hoc checksum, e.g. incorrect bit strings with checksum_ok=True.", "The lightweight checksum is useful as a plausibility check but too weak to be a reliable authenticator."),
        ("Near-correct fallback lock", "Fallback detections sometimes produced the correct payload bits even though no packet formally validated.", "The receiver can be close in time but still fail exact packet integrity, especially in the checksum tail."),
        ("Latency realism", "Successful detections at frame 0 were misleading before a minimum embedding offset was enforced.", "Unknown delay must be modeled explicitly or the sync problem looks easier than it really is."),
    ]

    for mode, symptom, interpretation in rows:
        cells = table.add_row().cells
        cells[0].text = mode
        cells[1].text = symptom
        cells[2].text = interpretation


def build_document():
    document = Document()
    set_page_margins(document.sections[0])
    style_base(document)

    p = document.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Audio Watermarking for Speech Through Telephony-Like Channels")

    p = document.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Final project report for EE123")

    meta = document.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(14)
    r = meta.add_run(f"Matthew Cullen | {date(2026, 5, 15).strftime('%B %d, %Y')}")
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x5A, 0x66, 0x73)

    add_body_paragraph(
        document,
        "This report summarizes the final state of the speech watermarking prototype implemented in 123Final.ipynb. "
        "The project goal was to embed an imperceptible signature into spoken audio, recover it after telephony-like distortion, "
        "and understand where synchronization and robustness fail when the method is pushed beyond favorable cases."
    )

    document.add_heading("Background", level=1)
    add_body_paragraph(
        document,
        "The project was motivated by a practical speech-authentication problem: can a low-power watermark be embedded into a human speech signal in a way that remains hard to hear but still survives a narrowband telephone-style channel? "
        "That requirement naturally points toward short-time spectral methods, because the communication path is dominated by resampling, bandlimiting, level changes, and quantization rather than perfect waveform preservation."
    )
    add_body_paragraph(
        document,
        "Two main references guided the design. Cox, Kilian, Leighton, and Shamoon's classic spread-spectrum watermarking paper [2] motivated the use of pseudorandom bipolar spreading over multiple spectral degrees of freedom rather than fragile single-bin marking. "
        "Huang and Ito's recent speech watermarking work [1] reinforced the idea that spectral embedding location matters and that speech-specific attacks or channel changes can defeat otherwise reasonable watermark choices if synchronization is not handled carefully."
    )
    add_body_paragraph(
        document,
        "Within that context, Barker codes were used as a compact synchronization tool. Barker sequences are attractive because they are short and have favorable autocorrelation sidelobes, which makes them a reasonable preamble choice when the audio clips themselves are only a few seconds long and longer pseudo-noise preambles would consume too much packet budget."
    )

    document.add_heading("Methods", level=1)
    add_body_paragraph(
        document,
        "The final prototype follows a packetized spread-spectrum workflow. Speech is normalized, split into overlapping frames, multiplied by a Hann window, transformed with the FFT, modified in selected bins, and then reconstructed by inverse FFT plus overlap-add."
    )
    add_small_heading(document, "Core signal processing")
    add_body_paragraph(
        document,
        "Framing used 20 ms windows with 10 ms hop. If x[n] is the input signal and w[n] is the Hann window, frame m is x_m[n] = x[n + mH] w[n], where H is the hop size. "
        "Each frame was transformed as X_m[k] = FFT{x_m[n]}, modified in magnitude over a selected band, then resynthesized with overlap-add."
    )
    add_small_heading(document, "Spread-spectrum embedding")
    add_body_paragraph(
        document,
        "Payload bits were mapped from {0,1} to {-1,+1}. For each symbol, a deterministic spreading code c[k] in {-1,+1} was regenerated from a shared seed and applied across selected frequency bins. "
        "The embedded magnitude perturbation followed the form |Y_m[k]| = |X_m[k]| + alpha s_m c[k] |X_m[k]|, where s_m is the bipolar watermark symbol and alpha is the embedding strength. "
        "Payload symbols used a base alpha, while preamble symbols used a stronger alpha to improve acquisition without raising audibility uniformly across the packet."
    )
    add_small_heading(document, "Packet structure and synchronization")
    add_body_paragraph(
        document,
        "The final protocol used a shared dictionary describing the preamble, payload length, repetition count, seed, and framing constants. "
        "A 13-bit Barker code served as the preamble. The payload itself was 8 bits, and each payload bit was repeated over 4 symbol positions. "
        "The decoder produced one soft frame score per symbol frame by correlating selected-bin magnitudes against the same spreading code, then slid the known preamble over that score stream with normalized correlation: corr[i] = <r_i, p> / (||r_i|| ||p||)."
    )
    add_body_paragraph(
        document,
        "Candidate starts were ranked by that normalized preamble correlation and then checked against the expected packet lattice. "
        "The current receiver confirms that a candidate is supported by the next expected packet position start + packet_symbol_len rather than trusting a single isolated peak. "
        "This is still only a partial synchronization solution, but it is substantially more realistic than the earlier oracle-assisted decoder."
    )
    add_small_heading(document, "Channel model")
    add_body_paragraph(
        document,
        "To approximate telephony distortion, the notebook supports bandlimiting to roughly 300-3400 Hz, resampling to 7.5 kHz, low-bit-depth quantization, and additive white Gaussian noise. "
        "A nonzero minimum embedding offset was also introduced so the receiver could not trivially succeed by assuming the watermark started at frame 0."
    )
    add_methods_table(document)

    document.add_heading("Metrics and Results", level=1)
    add_body_paragraph(
        document,
        "Performance was evaluated primarily with bit error rate (BER) on the 8-bit payload and with embedding signal-to-noise ratio for perceptual impact. "
        "The most polished demonstration cases were clean or lightly distorted speech files where the decoder localized the packet lattice and recovered the full payload correctly."
    )
    add_results_table(document)
    add_small_heading(document, "Interesting failure cases observed during debugging")
    add_failure_table(document)
    add_body_paragraph(
        document,
        "The Barker-13 preamble itself behaved well in isolation: the notebook's normalized autocorrelation plot showed a unit peak at zero lag and a largest sidelobe magnitude around 0.08, which is consistent with why Barker sequences are attractive for short synchronization words. "
        "However, good preamble autocorrelation alone did not guarantee robust packet recovery once speech-dependent host interference and channel distortion were introduced."
    )

    document.add_heading("Discussion", level=1)
    add_body_paragraph(
        document,
        "The strongest positive result is that the prototype now has a coherent end-to-end story: shared protocol definition, spread-spectrum embedding, packetized synchronization, telephony-style distortions, and BER measurement all exist in one working notebook. "
        "This is a meaningful improvement over the earlier stage where parts of the decoder still depended on embed-time oracle information."
    )
    add_body_paragraph(
        document,
        "The main shortcoming is synchronization. In the best cases, the receiver locks to the correct lattice and the payload decodes cleanly. In harder clips, the decoder produces too many plausible packet candidates, some of which are not on the transmitted lattice at all. "
        "That behavior indicates that the single-stream score sequence is still too influenced by host speech structure, especially in energy-deficient frames and at candidate positions that partially resemble the preamble by accident."
    )
    add_body_paragraph(
        document,
        "The ad hoc checksum also contributed to the mismatch between expectation and reality. It was useful as a lightweight plausibility filter, but it was not strong enough to reject all false locks. "
        "A CRC-8 variant was tested, and while it reduced false-valid packets, it also exposed a deeper problem: if the receiver does not first acquire the correct lattice, a stronger checksum simply leads to more rejections and more fallback behavior rather than better decoding."
    )
    add_body_paragraph(
        document,
        "The results were therefore somewhat mixed relative to the original ideal. The project succeeded at demonstrating an imperceptible short-time spectral watermark that can survive several realistic manipulations on some speech files, but it did not fully solve robust blind synchronization across all clips tested. "
        "That gap is most likely due to a combination of short-packet budget, variable speech energy, and the fact that compact audio clips leave only limited room to spend more symbols on synchronization or redundancy."
    )

    document.add_heading("Conclusion and Future Work", level=1)
    add_body_paragraph(
        document,
        "The project showed that spread-spectrum watermarking remains a strong foundation for speech watermarking under telephony-like distortion, especially when paired with repeated-bit payload support and a compact Barker-code preamble. "
        "It also made clear that synchronization, not basic embedding, is the dominant challenge in the blind-recovery setting."
    )
    add_bullets(
        document,
        [
            "Replace the ad hoc checksum with a stronger error-detection code once synchronization is stable enough to make that extra strictness useful.",
            "Upgrade the acquisition stage from single-peak correlation to a fuller lattice-lock procedure, potentially with local search around the predicted next packet position.",
            "Test a modest in-band carrier floor or pilot on weak frames so that silent or low-energy regions do not collapse the detector's confidence.",
            "Run a larger, cleaner BER table across more clips and distortion combinations so the final operating region is characterized quantitatively rather than anecdotally.",
            "Compare Barker-13 directly against another short pseudo-noise preamble of similar length to see whether the main bottleneck is the preamble itself or the downstream detector logic.",
        ],
    )
    add_body_paragraph(
        document,
        "The main lesson from the project is that a watermark can look strong in idealized embedding experiments and still fail once unknown delay and blind packet acquisition are required. "
        "The work therefore clarified both what was successful and where the true systems challenge lies: not in writing bits into speech, but in recovering them reliably when the channel and timing are no longer friendly."
    )

    document.add_heading("References", level=1)
    add_body_paragraph(
        document,
        "[1] X. Huang and A. Ito, \"Reversible Spectral Speech Watermarking with Variable Embedding Locations Against Spectrum-Based Attacks,\" Applied Sciences, vol. 15, no. 1, 2025."
    )
    add_body_paragraph(
        document,
        "[2] I. J. Cox, J. Kilian, F. T. Leighton, and T. Shamoon, \"Secure spread spectrum watermarking for multimedia,\" IEEE Transactions on Image Processing, vol. 6, no. 12, pp. 1673-1687, 1997."
    )

    section = document.add_section(WD_SECTION.NEW_PAGE)
    set_page_margins(section)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUT_PATH)


if __name__ == "__main__":
    build_document()
