// Convert a project Markdown report into a styled .docx (docx-js).
//
//   node scripts/report_tools/build_docx.js <input.md> <output.docx>
//
// Requires Node.js and the `docx` npm package (npm install docx). PDFs are made
// from the .docx with LibreOffice:  soffice --headless --convert-to pdf <file.docx>
//
// Optional environment variables:
//   DOC_TITLE, DOC_SUBTITLE   title page text (default: the first '# ' heading;
//                             a fully bold line right after it becomes the subtitle)
//   DOC_HEADER                running header text
//   DOC_CONTENTS_FIRST        extra first line of the contents list ('' for none)
//   DOC_H1_BREAK=0            sections flow on without a page break before each '# '
//
// Supported Markdown: headings, paragraphs, **bold**, *italic*, `code`, bullet and
// numbered lists, pipe tables, fenced code blocks, block quotes, images (paths are
// resolved relative to the Markdown file) and '---' rules.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat, ImageRun,
  Footer, Header, PageNumber,
} = require('docx');

const SRC = process.argv[2];
const OUT = process.argv[3];
const md = fs.readFileSync(SRC, 'utf8').split('\n');

const FONT = 'Arial';
const MONO = 'Courier New';
const CONTENT_W = 11906 - 2 * 1080; // A4 width minus 0.75in margins (DXA)
const ACCENT = '1F3864';
const ACCENT2 = '2E5597';

// ---------- inline formatting ----------
const LIT_STAR = '\u0001', LIT_PIPE = '\u0002', LIT_DOLLAR = '\u0003';
function unescape(s) {
  return s.replace(/\\\*/g, LIT_STAR).replace(/\\\|/g, LIT_PIPE).replace(/\\\$/g, LIT_DOLLAR);
}
function restore(s) {
  return s.split(LIT_STAR).join('*').split(LIT_PIPE).join('|').split(LIT_DOLLAR).join('$');
}
function inline(text, base = {}) {
  const runs = [];
  const re = /`([^`]+)`|\*\*(.+?)\*\*|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) runs.push(new TextRun({ text: restore(t), font: FONT, ...base, ...opts })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    if (m[1] !== undefined) {
      runs.push(new TextRun({ text: restore(m[1]), font: MONO, ...base, size: base.size ? base.size - 2 : 18, color: '8B1A1A' }));
    } else if (m[2] !== undefined) {
      runs.push(...inline(m[2], { ...base, bold: true }));
    } else if (m[3] !== undefined) {
      runs.push(...inline(m[3], { ...base, italics: true }));
    }
    last = m.index + m[0].length;
  }
  push(text.slice(last), {});
  return runs;
}

// ---------- block builders ----------
const children = [];
let numberInstance = 0;

function para(text, opts = {}) {
  const lines = text.split('\n');
  const runs = [];
  lines.forEach((ln, i) => {
    if (i > 0) runs.push(new TextRun({ text: '', break: 1 }));
    runs.push(...inline(unescape(ln), opts.run || {}));
  });
  return new Paragraph({ children: runs, spacing: { after: 100, line: 276 }, ...opts.para });
}

function codeBlock(lines) {
  return lines.map((ln, i) => new Paragraph({
    children: [new TextRun({ text: ln.length ? ln : ' ', font: MONO, size: 14 })],
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'F2F4F7' },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 120 : 0, line: 240 },
    indent: { left: 120, right: 120 },
    keepLines: true,
  }));
}

function quoteBlock(paras) {
  return paras.map((p, i) => new Paragraph({
    children: inline(unescape(p), { size: 20 }),
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'EEF3FA' },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT2, space: 8 } },
    indent: { left: 280, right: 200 },
    spacing: { before: i === 0 ? 120 : 60, after: 120, line: 276 },
  }));
}

function table(rows) {
  const split = (l) => {
    let s = unescape(l.trim());
    if (s.startsWith('|')) s = s.slice(1);
    if (s.endsWith('|')) s = s.slice(0, -1);
    return s.split('|').map((c) => c.trim());
  };
  const header = split(rows[0]);
  const body = rows.slice(2).map(split);
  const ncol = header.length;
  // column widths: every column at least as wide as its longest word; the rest by content length
  const clean = (t) => (t || '').replace(/\*\*|`/g, '');
  const CH = 92; // ~DXA per character at 8pt Arial (bold header slightly wider)
  const minW = [], want = [];
  for (let c = 0; c < ncol; c++) {
    const cells = [header[c], ...body.map((r) => r[c])].map(clean);
    const longestWord = Math.max(...cells.map((t) => Math.max(0, ...t.split(/\s+/).map((w) => w.length))));
    const longest = Math.max(...cells.map((t) => t.length));
    minW.push(Math.min((longestWord + 1) * CH + 170, CONTENT_W * 0.45));
    want.push(Math.min(Math.max(longest, 4), 60) * CH + 170);
  }
  let widths;
  const sumMin = minW.reduce((a, b) => a + b, 0);
  const sumWant = want.reduce((a, b) => a + b, 0);
  if (sumWant <= CONTENT_W) {
    widths = want.map((w) => w * CONTENT_W / sumWant);
  } else if (sumMin >= CONTENT_W) {
    widths = minW.map((w) => w * CONTENT_W / sumMin);
  } else {
    const extra = want.map((w, c) => Math.max(w - minW[c], 0));
    const se = extra.reduce((a, b) => a + b, 0) || 1;
    widths = minW.map((w, c) => w + extra[c] * (CONTENT_W - sumMin) / se);
  }
  widths = widths.map((w) => Math.floor(w));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFC7D5' };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mkRow = (cells, isHeader, idx) => new TableRow({
    tableHeader: isHeader,
    cantSplit: true,
    children: cells.map((txt, c) => new TableCell({
      width: { size: widths[c], type: WidthType.DXA },
      borders,
      shading: { type: ShadingType.CLEAR, color: 'auto', fill: isHeader ? 'DCE4F0' : (idx % 2 ? 'FFFFFF' : 'F7F9FC') },
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      children: [new Paragraph({ children: inline(txt || '', { size: 16, bold: isHeader || undefined }), spacing: { after: 0, line: 252 } })],
    })),
  });
  const trows = [mkRow(header, true, 0), ...body.map((r, i) => {
    while (r.length < ncol) r.push('');
    return mkRow(r.slice(0, ncol), false, i);
  })];
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trows }),
          new Paragraph({ children: [], spacing: { after: 80 } })];
}

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function image(caption, file) {
  const buf = fs.readFileSync(path.resolve(path.dirname(SRC), file));
  const { w, h } = pngSize(buf);
  const maxW = 600;
  const scale = Math.min(1, maxW / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: buf, transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 17, color: '555555', font: FONT })] }),
  ];
}

function hr() {
  return new Paragraph({ children: [], spacing: { before: 80, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: 'BFC7D5', space: 1 } } });
}

// ---------- collect H1 headings for the contents page ----------
const h1s = md.filter((l) => /^# /.test(l)).slice(1).map((l) => l.slice(2).trim());
const FIRST_H1 = (md.find((l) => /^# /.test(l)) || '# Report').slice(2).trim();
const TITLE = process.env.DOC_TITLE || FIRST_H1;
const SUBTITLE = process.env.DOC_SUBTITLE || '';
const H1_BREAK = process.env.DOC_H1_BREAK !== '0';
let expectSubtitle = false;
let firstSection = true;
const HEADER = process.env.DOC_HEADER || 'From Raw Data to Final Verdict  |  Kudrati Khoda';
const CONTENTS_FIRST = process.env.DOC_CONTENTS_FIRST === undefined ? 'How to read this document (labels used)' : process.env.DOC_CONTENTS_FIRST;

// ---------- main parse ----------
let i = 0;
let titleDone = false;
let contentsDone = false;
const nextNonEmpty = (k) => { while (k < md.length && md[k].trim() === '') k++; return md[k] || ''; };

while (i < md.length) {
  const line = md[i];
  if (line.startsWith('```')) {
    const buf = [];
    i++;
    while (i < md.length && !md[i].startsWith('```')) { buf.push(md[i]); i++; }
    i++;
    children.push(...codeBlock(buf));
    continue;
  }
  if (line.trim() === '') { i++; continue; }
  if (line.trim() === '---') {
    if (!nextNonEmpty(i + 1).startsWith('# ')) children.push(hr());
    i++; continue;
  }
  const img = line.match(/^!\[(.*)\]\((.*)\)\s*$/);
  if (img) { children.push(...image(img[1], img[2])); i++; continue; }

  if (line.startsWith('# ')) {
    const text = line.slice(2).trim();
    if (!titleDone) {
      titleDone = true;
      children.push(new Paragraph({ spacing: { before: 1800, after: 240 }, alignment: AlignmentType.LEFT,
        children: [new TextRun({ text: TITLE, bold: true, size: TITLE.length > 40 ? 44 : 52, color: ACCENT, font: FONT })] }));
      if (SUBTITLE) children.push(new Paragraph({ spacing: { after: 360 },
        children: [new TextRun({ text: SUBTITLE, size: 36, color: ACCENT2, font: FONT })] }));
      else expectSubtitle = true;
      children.push(new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT2, space: 4 } }, children: [], spacing: { after: 240 } }));
      i++; continue;
    }
    if (!contentsDone) {
      contentsDone = true;
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun({ text: 'Contents', font: FONT })] }));
      if (CONTENTS_FIRST) children.push(new Paragraph({ children: [new TextRun({ text: CONTENTS_FIRST, font: FONT, size: 20 })], spacing: { after: 40 } }));
      for (const h of h1s) children.push(new Paragraph({ children: [new TextRun({ text: h.replace(/—/g, '–'), font: FONT, size: 20 })], spacing: { after: 40 } }));
    }
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: H1_BREAK || firstSection, children: inline(unescape(text)) }));
    firstSection = false;
    i++; continue;
  }
  if (line.startsWith('#### ')) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_4, children: inline(unescape(line.slice(5))), keepNext: true })); i++; continue; }
  if (line.startsWith('### ')) { children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: inline(unescape(line.slice(4))), keepNext: true })); i++; continue; }
  if (line.startsWith('## ')) { const t3 = line.slice(3); children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: t3.startsWith('How to read'), children: inline(unescape(t3)), keepNext: true })); i++; continue; }

  if (line.startsWith('|')) {
    const rows = [];
    while (i < md.length && md[i].startsWith('|')) { rows.push(md[i]); i++; }
    children.push(...table(rows));
    continue;
  }
  if (line.startsWith('>')) {
    const paras = []; let cur = [];
    while (i < md.length && md[i].startsWith('>')) {
      const t = md[i].replace(/^>\s?/, '');
      if (t.trim() === '') { if (cur.length) paras.push(cur.join(' ')); cur = []; } else cur.push(t);
      i++;
    }
    if (cur.length) paras.push(cur.join(' '));
    children.push(...quoteBlock(paras));
    continue;
  }
  const bl = line.match(/^(\s*)[*-]\s+(.*)$/);
  if (bl) {
    const level = bl[1].length >= 2 ? 1 : 0;
    children.push(new Paragraph({ numbering: { reference: 'bullets', level }, children: inline(unescape(bl[2])), spacing: { after: 60, line: 264 } }));
    i++; continue;
  }
  const nl = line.match(/^(\s*)(\d+)\.\s+(.*)$/);
  if (nl) {
    numberInstance++;
    while (i < md.length) {
      const m2 = md[i].match(/^(\s*)(\d+)\.\s+(.*)$/);
      if (!m2) break;
      children.push(new Paragraph({ numbering: { reference: 'numbers', level: 0, instance: numberInstance }, children: inline(unescape(m2[3])), spacing: { after: 60, line: 264 } }));
      i++;
    }
    continue;
  }
  // plain paragraph: join following plain lines with line breaks
  const sub = expectSubtitle && line.match(/^\*\*(.+)\*\*\s*$/);
  expectSubtitle = false;
  if (sub) {
    children.push(new Paragraph({ spacing: { after: 360 }, children: [new TextRun({ text: restore(unescape(sub[1])), size: 30, color: ACCENT2, font: FONT })] }));
    i++; continue;
  }
  const buf = [line];
  i++;
  while (i < md.length && md[i].trim() !== '' && !/^(#|\||>|```|!\[|\s*[*-]\s|\s*\d+\.\s|---)/.test(md[i])) { buf.push(md[i]); i++; }
  children.push(para(buf.join('\n')));
}

// ---------- document ----------
const doc = new Document({
  creator: 'Kudrati Khoda',
  title: TITLE + ' - ' + SUBTITLE,
  description: 'Teaching and interview-defence guide reverse-engineered from the KK repository',
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 120, after: 200 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT2, space: 4 } } } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, color: ACCENT2, font: FONT },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
      { id: 'Heading4', name: 'Heading 4', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 20, bold: true, italics: true, color: ACCENT, font: FONT },
        paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 3 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '–', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 800, hanging: 260 } } } },
      ] },
      { reference: 'numbers', levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 440, hanging: 300 } } } },
      ] },
    ],
  },
  sections: [{
    properties: { page: { margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: HEADER, size: 16, color: '7A7A7A', font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: '7A7A7A', font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT, b.length, 'bytes;', children.length, 'blocks'); });
