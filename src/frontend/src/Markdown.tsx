/**
 * Markdown renderer for assistant answers.
 *
 * Wraps ``react-markdown`` + ``remark-gfm`` so LLM responses render
 * with proper structure (headings, lists, tables, code blocks, bold /
 * italic, links) instead of being dumped as raw text. The pre-existing
 * ``tokenizeCitations`` behavior is preserved by a custom remark
 * plugin that converts ``[n]`` markers into ``<span class="citation-
 * chip" data-citation-index="n">`` elements before markdown renders.
 *
 * Why a plugin, not a post-process step:
 *
 * - Splitting the content by ``[n]`` first and running markdown on
 *   each chunk breaks markdown structure (a marker inside a list item
 *   or table cell gets split into its own block).
 * - Tokenizing inside the AST keeps citation chips inline with the
 *   surrounding markdown: bold spanning a citation, citations inside
 *   list items, etc. all render correctly.
 *
 * Web citations (``source_kind === "web"`` with a URL) render as
 * click-through links opening in a new tab, just like the chip row
 * below the answer used to do. Local citations render as clickable
 * spans that fire ``rag:citation-click`` so the source panel can
 * scroll/highlight the matching card.
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit, SKIP } from "unist-util-visit";
import type { Source } from "./chat-utils";
import { CitationChip } from "./components/CitationChip";

interface Props {
  content: string;
  sources?: Source[];
}

/** Custom remark plugin: split ``text`` nodes so each ``[n]`` marker
 *  becomes a synthetic ``link`` node with citation-chip class.
 *  ``mdast-util-to-hast`` then emits an ``<a>`` element that
 *  react-markdown routes to our override below. */
function remarkCitations() {
  return (tree: any) => {
    visit(tree, "text", (node: any, index, parent) => {
      if (!parent || typeof index !== "number") return;
      const text: string = node.value || "";
      const re = /\[(\d+)\]/g;
      const matches: Array<{ start: number; end: number; idx: number }> = [];
      let m: RegExpExecArray | null;
      while ((m = re.exec(text)) !== null) {
        matches.push({
          start: m.index,
          end: m.index + m[0].length,
          idx: Number(m[1]),
        });
      }
      if (!matches.length) return;
      const newNodes: any[] = [];
      let cursor = 0;
      for (const mt of matches) {
        if (mt.start > cursor) {
          newNodes.push({ type: "text", value: text.slice(cursor, mt.start) });
        }
        // ``link`` MDAST node → emits <a> in HAST, which we override
        // via react-markdown's components.a below. hProperties carries
        // our chip class + data attribute into the rendered element so
        // our override can identify and re-render it as a span.
        newNodes.push({
          type: "link",
          url: `#cite-${mt.idx}`,
          title: null,
          data: {
            hProperties: {
              className: "citation-chip",
              "data-citation-index": String(mt.idx),
            },
          },
          children: [{ type: "text", value: `[${mt.idx}]` }],
        });
        cursor = mt.end;
      }
      if (cursor < text.length) {
        newNodes.push({ type: "text", value: text.slice(cursor) });
      }
      parent.children.splice(index, 1, ...newNodes);
      return [SKIP, index + newNodes.length];
    });
  };
}

export function Markdown({ content, sources }: Props) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkCitations]}
        components={{
          // Override <a>: if our remark plugin flagged it as a citation
          // chip (via className), render a clickable <span> instead.
          // Otherwise render a normal link.
          a({ node, className, children, ...props }) {
            const isCitation = (className || "").includes("citation-chip");
            if (isCitation) {
              const idx = Number(
                node?.properties?.["data-citation-index"] ??
                  // hProperties may surface as plain string on the
                  // rendered element — fallback either way.
                  (node as any)?.data?.hProperties?.["data-citation-index"],
              );
              const source = sources?.find((s) => s.index === idx);
              const isWeb =
                source?.source_kind === "web" && !!source?.url;
              if (isWeb) {
                return (
                  <a
                    className="citation-chip-link"
                    href={source!.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    title={source!.text}
                  >
                    [{idx}] {source!.domain || source!.url}
                  </a>
                );
              }
              return (
                <CitationChip idx={idx} source={source} />
              );
            }
            return (
              <a
                {...props}
                className={className}
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}