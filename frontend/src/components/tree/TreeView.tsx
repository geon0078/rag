import { useNavigate, useParams } from "react-router-dom";
import { useTree, type TreeNode } from "@/api/tree";

// 운영웹통합명세서 §6.3 — Sidebar 트리.
// Day 3 구현: 단순 nested <ul> (react-arborist 가상화는 데이터가 더 커지면 도입).
export default function TreeView() {
  const { collection } = useParams<{ collection: string }>();
  const { data, isLoading, error } = useTree(collection);

  if (isLoading) return <p className="text-sm text-zinc-500">트리 로딩 중…</p>;
  if (error)
    return (
      <p className="text-sm text-red-600">
        트리 로드 실패: {(error as Error).message}
      </p>
    );
  if (!data) return null;

  return (
    <div className="text-sm">
      {!collection && (
        <p className="mb-2 text-xs text-zinc-500">
          컬렉션 선택 ({data.count}개 청크)
        </p>
      )}
      <ul className="space-y-0.5">
        {data.tree.map((n) => (
          <Node key={n.id} node={n} indent={0} />
        ))}
      </ul>
    </div>
  );
}

function Node({ node, indent }: { node: TreeNode; indent: number }) {
  const navigate = useNavigate();
  const hasChildren = node.children && node.children.length > 0;
  const isCollectionRoot = indent === 0 && !node.doc_id && !hasChildren;

  const onClick = () => {
    if (node.doc_id) {
      navigate(`/chunks/${encodeURIComponent(node.doc_id)}`);
    } else if (isCollectionRoot) {
      navigate(`/collections/${encodeURIComponent(node.name)}`);
    }
  };

  const icon = indent === 0 ? "📚" : node.doc_id ? "📋" : "📁";
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={
          "flex w-full items-center gap-1.5 truncate rounded-md py-1 pr-2 text-left transition-colors hover:bg-zinc-100 " +
          (node.doc_id ? "text-zinc-700" : "font-medium text-zinc-800")
        }
        style={{ paddingLeft: 6 + indent * 12 }}
        title={node.name}
      >
        <span className="text-xs leading-none">{icon}</span>
        <span className="flex-1 truncate text-sm">{node.name}</span>
        {node.count !== undefined && (
          <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500">
            {node.count.toLocaleString()}
          </span>
        )}
      </button>
      {hasChildren && (
        <ul className="space-y-0.5">
          {node.children.map((c) => (
            <Node key={c.id} node={c} indent={indent + 1} />
          ))}
        </ul>
      )}
    </li>
  );
}
