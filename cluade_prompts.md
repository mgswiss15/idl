Here are the condensed versions:

---

**SLIDES PROMPT:**

Generate Beamer slides for Session X of an Introduction to Deep Learning course at THWS. Audience: first-semester Masters students with CS Bachelor background.

Discuss content and structure before writing any LaTeX. Generate section by section and wait for feedback.

BEAMER SETUP:
\documentclass[compress, xcolor={svgnames}, aspectratio=169, smaller, t]{beamer}
- \vskip 2em after every frame title
- No articles in frame titles, no overlays, no slide notes blocks
- Section divider: empty frame with \vskip 8em and {\Large \structure{Title}}
- \structure{} for emphasis, \alert{} for warnings/risks
- Sparse slides: one idea per slide, bullet points max one sentence
- Code: {\scriptsize\ttfamily ...} with manual line breaks
- Highlights: {\setlength{\fboxsep}{8pt}\colorbox{thwsorange!10}{\structure{message}}}
- Citations: plain author-year with \href{arxiv url}{Author et al., Year} — never \parencite in frames

COLORS: thwsblue (22,60,105), carnelian (179,27,27), thwsorange (255,106,0), thwspetrol (0,85,100), thwsgrey (217,217,217), lightblue (166,189,219)

FIGURES: \tikzsetnextfilename{} before every TikZ \input; suggest TikZ figures and provide code separately; images at \CommonPath/Pics/, TikZ at \CommonPath/tikz/

MATH: \vx \vy vectors; \pvec for \boldsymbol{\theta}; \thetamat{l} weights; \thetavec{l}_0 bias; \act activation; \kernel filters; \mX input; \mW weights; bias-before-weight convention; 0-based layer indexing

OUTPUT: standalone .tex section (no preamble) + separate .bib file in biblatex/biber format

---

**LECTURE NOTES PROMPT:**

Generate lecture notes for Session X of an Introduction to Deep Learning course at THWS from the attached slides. The notes are a standalone section included via \input{} in a scrartcl document — no preamble or \begin{document}.

Generate section by section and wait for feedback.

STRUCTURE:
- \section{}, \subsection{}, \subsubsection{} matching slide sections
- \slideref{} at the start of each subsection listing relevant slide titles
- Short bridging paragraph between \subsection{} and first \subsubsection{}

STYLE:
- Self-contained narrative prose readable without the slides
- Formal definitions with accessible explanation — no bullet points
- Use -- for dashes, never ---
- \textbf{} for emphasis, \parencite{} and \textcite{} for citations
- \href{} for clickable links, verbatim for code blocks
- Concise figure captions, no horizontal separation in itemize lists

FIGURES: wrapfigure environments close to relevant text; \CommonPath/Pics/ for images; \CommonPath/tikz/ for TikZ; \tikzsetnextfilename{} before every TikZ \input

CONTENT: expand slide bullets into reasoned prose; add intuition and motivation; do not reproduce slides verbatim — the notes must add value

OUTPUT: standalone .tex section + separate .bib file in biblatex/biber format
