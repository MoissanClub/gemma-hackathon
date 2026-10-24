# Offline judge presentation

Open `index.html` directly in Chrome, Safari, or Edge. No server, installation, internet connection, or external assets are needed.

The single HTML file contains 12 main slides, 2 evidence appendices, photos, animated charts, speaker notes, and sources. It can be copied to another computer by itself.

## Controls

| Key | Action |
| --- | --- |
| Right / Space / Page Down | Next slide |
| Left / Page Up | Previous slide |
| F | Full screen |
| O | Slide overview |
| N | Speaker notes and evidence |
| C | Switch the model-size perception chart |
| R | Replay the current slide animation |
| B or period | Black screen |
| M | Reduce motion |
| E | Edit the AGX Thor task placeholder |
| Escape | Close notes/overview or finish editing |
| Home / End | Opening / closing slide |
| A | Benchmark appendix |
| ? | Help |

Click the screen's left or right edge to navigate, or swipe on a touchscreen. The slide number is encoded in the URL hash.

## Before presenting

On the 12B/Thor slide, press **E** and replace the yellow task placeholder. The edit persists in the current browser if local storage is available; it does not rewrite the HTML file. To make the change portable, edit the `data-edit="thor-task"` element in the HTML itself.

Sources and scope are embedded in each slide's **N** panel. Model-size charts use the same-day October 22 Orin measurements. Code-generation times use complete-request wall medians from the saved experiment summaries. The LiteRT comparison is native spoken target choice, not complete robot pick time.

PiPER, mobile SO-101, AGX Thor, and the latest G1 third-person automatic-calibration updates include the presenter's latest team reports. No unrecorded performance numbers are supplied for them.

Browser printing exposes all 14 slides. The file has no automatic external requests and uses system fonts.
