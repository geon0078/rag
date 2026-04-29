import { Routes, Route } from "react-router-dom";
import AppLayout from "./components/layout/AppLayout";
import HomePage from "./pages/HomePage";
import ChunkPage from "./pages/ChunkPage";
import CollectionPage from "./pages/CollectionPage";
import PreviewPage from "./pages/PreviewPage";
import IndexingPage from "./pages/IndexingPage";
import FaqNewPage from "./pages/FaqNewPage";
import HistoryPage from "./pages/HistoryPage";
import UploadPage from "./pages/UploadPage";
import ChatPage from "./pages/ChatPage";

// 운영웹통합명세서 §10.2 + onyx+docmost 개발.md §5.
// 학생 chat (사이드바 없음): "/chat" — AppLayout 외부
// 운영자 admin (사이드바 포함): "/" 하위 — AppLayout
export default function App() {
  return (
    <Routes>
      {/* 학생용 — AppLayout 외부에 mount (admin nav 노출 안 함) */}
      <Route path="/chat" element={<ChatPage />} />

      {/* 운영자용 — admin sidebar + 페이지들 */}
      <Route path="/" element={<AppLayout />}>
        <Route index element={<HomePage />} />
        <Route path="collections/:name" element={<CollectionPage />} />
        <Route path="chunks/:doc_id" element={<ChunkPage />} />
        <Route path="preview" element={<PreviewPage />} />
        <Route path="indexing" element={<IndexingPage />} />
        <Route path="faq/new" element={<FaqNewPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="upload" element={<UploadPage />} />
      </Route>
    </Routes>
  );
}
