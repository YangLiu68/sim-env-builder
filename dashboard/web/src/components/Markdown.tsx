import DOMPurify from "dompurify";
import { marked } from "marked";
import { useMemo } from "react";

marked.setOptions({ gfm: true, breaks: false });

export default function Markdown({ text, className = "" }: { text: string; className?: string }) {
  const html = useMemo(() => DOMPurify.sanitize(marked.parse(text || "", { async: false }) as string), [text]);
  return <div className={`md ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
