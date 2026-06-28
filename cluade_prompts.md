> Prompt for slides

Here it is:

---

I am preparing slides for Session X of my "Introduction to Deep Learning" course at THWS. The audience is first-semester Masters students with a CS Bachelor background. Please generate the slides following these conventions.

DOCUMENT STRUCTURE:
- Use \section{} for main topics and \begin{frame}{Title} for individual slides
- Include a \begin{frame}{Lecture Overview} \tableofcontents \end{frame} after the title slide
- Use a plain section divider slide between major sections:
  \begin{frame}{} \vskip 8em \begin{center} {\Large \structure{Section Title}} \end{center} \end{frame}

BEAMER CONVENTIONS:
- Document class: \documentclass[compress, xcolor={svgnames}, aspectratio=169, smaller, t]{beamer}
- Always add \vskip 2em after frame titles
- No articles in frame titles (write "Convolutional Layers" not "The Convolutional Layers")
- No overlays or animations
- Use \structure{} for emphasis and key terms
- Use \alert{} for warnings, risks, and important caveats
- Use FontAwesome5 icons where appropriate (pdfLaTeX compatible)
- No \begin{block} environments — use \structure{} and \alert{} inline instead

LAYOUT:
- Use \begin{columns} for side-by-side content
- Keep slides sparse — one idea per slide, no walls of text
- Use short bullet points of at most one sentence each
- Code snippets use {\scriptsize\ttfamily ...} with manual line breaks
- Highlighted practical implications use \colorbox with increased \fboxsep:
  {\setlength{\fboxsep}{8pt}\colorbox{thwsorange!10}{\structure{key message}}}

FIGURES AND TIKZ:
- Use \includegraphics for images, with path \CommonPath/Pics/
- Use \input for TikZ figures, with path \CommonPath/tikz/
- Always place \tikzsetnextfilename{filename} before every TikZ \input
- Suggest TikZ figures where a diagram would aid understanding, and provide the TikZ code separately

CITATIONS:
- Do not use \parencite or \textcite inside Beamer frames — these do not work
- Use plain author-year with \href pointing to the paper URL instead:
  \href{https://arxiv.org/abs/...}{Author et al., Year}

COLOR SCHEME:
- \structure{} renders in thwsblue (RGB 22, 60, 105)
- \alert{} renders in carnelian (RGB 179, 27, 27)
- thwsorange (RGB 255, 106, 0) for highlights
- thwspetrol (RGB 0, 85, 100) for frame titles
- thwsgrey (RGB 217, 217, 217) for secondary elements
- lightblue (RGB 166, 189, 219) for supplementary content

MATH CONVENTIONS:
- Use \vx, \vy for vectors; \pvec for parameter vector \boldsymbol{\theta}
- Use \thetamat{l} for weight matrix at layer l; \thetavec{l}_0 for bias
- Use \act for activation function; \kernel for CNN filters \boldsymbol{\kappa}
- Use \mX for input matrix; \mW for weight matrix
- Bias-before-weight convention: \thetavec{l}_0 + \thetamat{l}\vh^{(l-1)}
- 0-based layer superscript indexing

CONTENT GUIDELINES:
- Storyline first — each slide should have a clear narrative purpose
- Motivate before introducing — explain why before explaining what
- One concrete example per major concept
- Practical PyTorch code where relevant, using nn.Module style
- Forward references to later sessions are acceptable but should be clearly marked as previews
- Do not include slide notes blocks

BIBLIOGRAPHY:
- Provide a separate .bib file for all cited works
- Use biblatex/biber format

Please discuss the content and structure with me before writing any LaTeX. Generate the slides section by section and wait for my feedback before proceeding to the next section.

---

> Prompt for Lecture notes

---

I am attaching the slides for Session X of my "Introduction to Deep Learning" course at THWS. Please generate lecture notes from these slides following these conventions.

The output is a standalone section to be included in the main notes document via \input{} — do not include a document preamble, \begin{document}, or \end{document}.

DOCUMENT STRUCTURE:
- Use \section{}, \subsection{}, and \subsubsection{} matching the slide sections
- Begin each subsection with a \slideref{} command listing the relevant slide titles
- Write a short bridging paragraph between \subsection{} and the first \subsubsection{} where needed

WRITING STYLE:
- Self-contained narrative prose — the notes should be readable without the slides
- Mix of formal definitions and accessible explanation
- No bullet points — convert all slide bullet points into flowing paragraphs
- Avoid --- for dashes, use -- instead
- Concise figure captions
- No horizontal separation in itemize lists

FIGURES:
- Include all figures from the slides using \includegraphics or \input for TikZ
- Place figures as wrapfigure environments close to the relevant text
- Use \CommonPath/Pics/ for images and \CommonPath/tikz/ for TikZ figures
- Use \tikzsetnextfilename{} before every TikZ \input

LATEX CONVENTIONS:
- Document class is scrartcl
- Use \parencite{} and \textcite{} for citations
- Use \textbf{} for emphasis (not \structure{} which is Beamer-only)
- Use verbatim environments for code blocks
- Normalisation statistics, hyperlinks, and clickable URLs should be included where the slides contain them
- Use \href{} for clickable links

CONTENT GUIDELINES:
- Expand on slide bullet points — explain the reasoning, not just the facts
- Add intuition and motivation where the slides are concise
- Highlight practical implications clearly
- Where the slides reference a figure or diagram, describe what it shows and what the student should take away from it
- Do not reproduce the slide content verbatim — the notes should add value

BIBLIOGRAPHY:
- Provide a separate .bib file containing all references cited in the notes
- Use biblatex/biber format
- Include all papers referenced in the slides plus any additional sources added in the notes

Please generate the notes section by section and wait for my feedback before proceeding to the next section.

---