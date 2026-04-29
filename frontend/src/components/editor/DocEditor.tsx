import { useEditor, EditorContent } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Placeholder from "@tiptap/extension-placeholder";
import { useEffect, useRef } from "react";

// 운영웹통합명세서 §6.4 (편집 모드) + §10.3 — TipTap 위지윅 편집기.
// debounce 1초 자동 저장, value 변경 시 onSave 콜백 호출.
export default function DocEditor({
  initialContent,
  onSave,
}: {
  initialContent: string;
  onSave: (next: string) => void | Promise<void>;
}) {
  const lastSaved = useRef(initialContent);
  const editor = useEditor({
    extensions: [
      StarterKit,
      Placeholder.configure({ placeholder: "내용을 입력하세요…" }),
    ],
    content: initialContent,
    editorProps: {
      attributes: {
        class:
          "prose max-w-none min-h-[300px] focus:outline-none rounded border border-zinc-200 bg-white p-3",
      },
    },
  });

  useEffect(() => {
    if (!editor) return;
    const timer = setInterval(() => {
      const html = editor.getHTML();
      if (html !== lastSaved.current) {
        lastSaved.current = html;
        void onSave(html);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [editor, onSave]);

  if (!editor) return null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-1 text-xs">
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleBold().run()}
          className={
            "rounded border px-2 py-1 " +
            (editor.isActive("bold")
              ? "border-zinc-900 bg-zinc-900 text-white"
              : "border-zinc-300 bg-white")
          }
        >
          B
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleItalic().run()}
          className={
            "rounded border px-2 py-1 italic " +
            (editor.isActive("italic")
              ? "border-zinc-900 bg-zinc-900 text-white"
              : "border-zinc-300 bg-white")
          }
        >
          I
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
          className={
            "rounded border px-2 py-1 " +
            (editor.isActive("heading", { level: 2 })
              ? "border-zinc-900 bg-zinc-900 text-white"
              : "border-zinc-300 bg-white")
          }
        >
          H2
        </button>
        <button
          type="button"
          onClick={() => editor.chain().focus().toggleBulletList().run()}
          className={
            "rounded border px-2 py-1 " +
            (editor.isActive("bulletList")
              ? "border-zinc-900 bg-zinc-900 text-white"
              : "border-zinc-300 bg-white")
          }
        >
          • List
        </button>
        <span className="ml-auto text-zinc-400">자동 저장 (1초)</span>
      </div>
      <EditorContent editor={editor} />
    </div>
  );
}
