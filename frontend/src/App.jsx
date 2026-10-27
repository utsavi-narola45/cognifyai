import { useEffect, useRef, useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Plus, Upload, FileText, ArrowUp, Trash2, PanelLeft, LoaderCircle,
  MessageSquare, Search, X, AlertCircle, CheckCircle2, Pencil, Check,
  Copy, RefreshCw, Square, ArrowDown, UploadCloud, Sun, Moon,
  Library, History, Layers, BookOpenText, ListChecks, GitCompare,
  Image as ImageIcon, Download, ChevronLeft, ChevronRight,
} from "lucide-react";
import "./index.css";

const API = "http://127.0.0.1:8000";

const SUGGESTIONS = [
  {
    icon: BookOpenText,
    title: "Summarize",
    text: "Summarize the key ideas across my documents.",
  },
  {
    icon: ListChecks,
    title: "Extract facts",
    text: "What are the important facts, names and figures in my documents?",
  },
  {
    icon: GitCompare,
    title: "Compare",
    text: "Compare the main ideas of my documents.",
  },
  {
    icon: Layers,
    title: "Inventory",
    text: "Which documents do I have and what is each about?",
  },
];

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// "[Source 2]" / "[Source 1, Source 3]" / "[Source 1: file.pdf]"
// become one clickable citation chip per source number.
function linkCitations(text) {
  return text.replace(/\[([^\]\n]*Source[^\]\n]*)\]/gi, (match, inner) => {
    const numbers = [...inner.matchAll(/Source\s*(\d+)/gi)].map((m) => m[1]);
    if (!numbers.length) return match;
    return [...new Set(numbers)]
      .map((n) => `[${n}](#src-${n})`)
      .join(" ");
  });
}

function Logo({ size = 34 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="12" fill="url(#cg)" />
      <g stroke="#fff" strokeWidth="2" strokeLinecap="round" fill="none">
        <path d="M13 24l7-9 7 9" />
        <path d="M20 15v-3" />
      </g>
      <g fill="#fff">
        <circle cx="13" cy="25" r="2.6" />
        <circle cx="27" cy="25" r="2.6" />
        <circle cx="20" cy="11.5" r="2.6" />
      </g>
    </svg>
  );
}

function readTheme() {
  try {
    return localStorage.getItem("cognify-theme") || "dark";
  } catch {
    return "dark";
  }
}

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [conversationId, setConversationId] = useState(null);
  const [selectedDocs, setSelectedDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [tab, setTab] = useState("library");
  const [search, setSearch] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);
  const [editText, setEditText] = useState("");
  const [evidence, setEvidence] = useState(null); // {index, focus}
  const [conversations, setConversations] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(null);
  const [showJump, setShowJump] = useState(false);
  const [theme, setTheme] = useState(readTheme);
  const [lightbox, setLightbox] = useState(null); // {images, index}

  const abortRef = useRef(null);
  const chatRef = useRef(null);
  const stickRef = useRef(true);
  const bottomRef = useRef(null);
  const fileRef = useRef(null);
  const composerRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("cognify-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e) => {
      if (e.key === "Escape") setLightbox(null);
      if (e.key === "ArrowRight")
        setLightbox((l) => l && { ...l, index: (l.index + 1) % l.images.length });
      if (e.key === "ArrowLeft")
        setLightbox((l) => l && {
          ...l, index: (l.index - 1 + l.images.length) % l.images.length,
        });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  async function loadDocuments() {
    try {
      const response = await axios.get(`${API}/api/documents`);
      setDocuments(response.data.documents);
      setApiReady(true);
    } catch {
      setApiReady(false);
      setError("Cannot reach the CognifyAI backend. Is it running on port 8000?");
    }
  }

  async function loadConversations() {
    try {
      const response = await axios.get(`${API}/api/conversations`);
      setConversations(response.data.conversations);
    } catch {
      /* history is optional */
    }
  }

  useEffect(() => {
    loadDocuments();
    loadConversations();
  }, []);

  useEffect(() => {
    if (stickRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "auto" });
    }
  }, [messages]);

  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [question]);

  function onChatScroll() {
    const el = chatRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    stickRef.current = atBottom;
    setShowJump(!atBottom);
  }

  function jumpToBottom() {
    stickRef.current = true;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;

    setUploading(true);
    setError("");
    setNotice("");
    setTab("library");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await axios.post(`${API}/api/documents/upload`, formData);
      const added = response.data.documents || [];
      const failures = response.data.errors || [];

      await loadDocuments();

      if (added.length) {
        setNotice(`${added.length} document(s) added to your library.`);
      }
      if (failures.length) {
        setError(failures.map((item) => `${item.name}: ${item.error}`).join(" | "));
      }
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => `${d.name}: ${d.error}`).join(" | ")
            : "Upload failed. Check the file type and backend logs."
      );
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function deleteDocument(id) {
    try {
      await axios.delete(`${API}/api/documents/${id}`);
      setDocuments((previous) => previous.filter((doc) => doc.id !== id));
      setSelectedDocs((previous) => previous.filter((docId) => docId !== id));
      setNotice("Document removed.");
      setError("");
    } catch {
      setError("Could not delete the document.");
    }
  }

  function newChat() {
    abortRef.current?.abort();
    setConversationId(null);
    setEditingIndex(null);
    setEvidence(null);
    setMessages([]);
    setQuestion("");
    setError("");
    setNotice("");
  }

  async function ask(text, editIndex = null) {
    if (!text || sending) return;

    const editing = editIndex !== null;
    const truncateId = editing ? messages[editIndex]?.id ?? null : null;
    const base = editing ? messages.slice(0, editIndex) : messages;

    setMessages([
      ...base,
      { role: "user", content: text },
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setEditingIndex(null);
    setEvidence(null);
    setQuestion("");
    setSending(true);
    setError("");
    setNotice("");
    stickRef.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    const patchLast = (patch) =>
      setMessages((previous) => {
        const next = [...previous];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });

    let acc = "";
    let finished = false;

    try {
      const response = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question: text,
          conversation_id: conversationId,
          document_ids: selectedDocs.length ? selectedDocs : null,
          truncate_from_id: truncateId,
        }),
      });

      if (!response.ok) {
        let detail = "";
        try {
          detail = (await response.json()).detail;
        } catch {
          /* not json */
        }
        throw new Error(
          typeof detail === "string" && detail ? detail : "Request failed"
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handle = (event) => {
        if (event.type === "meta") {
          setConversationId(event.conversation_id);
          patchLast({
            sources: event.sources,
            images: event.images || [],
            mode: event.mode,
          });
        } else if (event.type === "token") {
          acc += event.text;
          patchLast({ content: acc });
        } else if (event.type === "done") {
          finished = true;
          setMessages((previous) => {
            const next = [...previous];
            const last = next.length - 1;
            next[last] = {
              ...next[last],
              content: event.answer,
              id: event.assistant_message_id,
              streaming: false,
            };
            next[last - 1] = { ...next[last - 1], id: event.user_message_id };
            return next;
          });
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      };

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (line.trim()) handle(JSON.parse(line));
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        patchLast({
          content: acc ? `${acc}\n\n*(stopped)*` : "*(stopped)*",
          streaming: false,
        });
      } else {
        patchLast({
          content:
            err.message ||
            "I couldn't generate an answer. Check that the backend and Ollama are running.",
          sources: [],
          failed: true,
          streaming: false,
        });
      }
    } finally {
      if (!finished) patchLast({ streaming: false });
      abortRef.current = null;
      setSending(false);
      loadConversations();
    }
  }

  function sendQuestion(event) {
    event?.preventDefault();
    ask(question.trim());
  }

  function regenerate(index) {
    const userIndex = index - 1;
    if (userIndex < 0 || sending) return;
    ask(messages[userIndex].content, userIndex);
  }

  async function copyMessage(index) {
    try {
      await navigator.clipboard.writeText(messages[index].content);
      setCopied(index);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      setError("Could not copy to the clipboard.");
    }
  }

  function startEdit(index) {
    setEditingIndex(index);
    setEditText(messages[index].content);
  }

  function openEvidence(index, focus = null) {
    setEvidence({ index, focus });
    if (focus) {
      setTimeout(() => {
        document
          .getElementById(`ev-${focus}`)
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 80);
    }
  }

  async function openConversation(id) {
    if (sending || id === conversationId) return;

    try {
      const response = await axios.get(`${API}/api/conversations/${id}`);
      setConversationId(id);
      setEditingIndex(null);
      setEvidence(null);
      setError("");
      setNotice("");
      setMessages(
        response.data.messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          sources: message.sources,
          images: message.images || [],
          mode: message.sources?.length ? "document" : "casual",
        }))
      );
      stickRef.current = true;
    } catch {
      setError("Could not load that conversation.");
    }
  }

  async function deleteConversation(id) {
    try {
      await axios.delete(`${API}/api/conversations/${id}`);
      if (id === conversationId) newChat();
      loadConversations();
    } catch {
      setError("Could not delete the conversation.");
    }
  }

  const visibleDocuments = documents.filter((doc) =>
    doc.name.toLowerCase().includes(search.toLowerCase())
  );

  const evidenceMessage = evidence ? messages[evidence.index] : null;
  const evidenceDocs = evidenceMessage
    ? new Set(evidenceMessage.sources.map((s) => s.document_name)).size
    : 0;

  const scopeLabel = selectedDocs.length
    ? `${selectedDocs.length} selected`
    : documents.length
      ? `All ${documents.length} documents`
      : "No documents yet";

  return (
    <div
      className={`shell ${sidebarOpen ? "" : "shell-collapsed"} ${evidence ? "shell-evidence" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!dragging) setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        uploadFiles(e.dataTransfer.files);
      }}
    >
      {dragging && (
        <div className="drop-overlay">
          <UploadCloud size={46} />
          <strong>Drop to add to your library</strong>
          <span>PDF, DOCX or TXT · up to 15 MB each</span>
        </div>
      )}

      {/* ------------------------------ Sidebar ------------------------------ */}
      <aside className="side">
        <div className="side-top">
          <div className="brand">
            <Logo />
            <div>
              <h1>
                Cognify<span>AI</span>
              </h1>
              <p>Talk to your documents</p>
            </div>
          </div>

          <button className="new-chat" onClick={newChat}>
            <Plus size={17} /> New chat
          </button>

          <div className="tabs" role="tablist">
            <button
              role="tab"
              aria-selected={tab === "library"}
              className={tab === "library" ? "on" : ""}
              onClick={() => setTab("library")}
            >
              <Library size={14} /> Library
              <em>{documents.length}</em>
            </button>
            <button
              role="tab"
              aria-selected={tab === "history"}
              className={tab === "history" ? "on" : ""}
              onClick={() => setTab("history")}
            >
              <History size={14} /> History
              <em>{conversations.length}</em>
            </button>
          </div>
        </div>

        <div className="side-body">
          {tab === "library" ? (
            <>
              <label className="search">
                <Search size={14} />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Filter documents"
                />
              </label>

              <div className="scope-hint">
                {selectedDocs.length ? (
                  <>
                    Chatting with {selectedDocs.length} selected
                    <button onClick={() => setSelectedDocs([])}>Clear</button>
                  </>
                ) : (
                  "Tick documents to focus the chat on them"
                )}
              </div>

              <div className="doc-list">
                {visibleDocuments.map((doc) => {
                  const on = selectedDocs.includes(doc.id);
                  return (
                    <div className={`doc ${on ? "on" : ""}`} key={doc.id}>
                      <label>
                        <input
                          type="checkbox"
                          checked={on}
                          onChange={(e) =>
                            setSelectedDocs((previous) =>
                              e.target.checked
                                ? [...previous, doc.id]
                                : previous.filter((id) => id !== doc.id)
                            )
                          }
                        />
                        <span className="doc-icon">
                          <FileText size={16} />
                        </span>
                        <span className="doc-meta">
                          <b title={doc.name}>{doc.name}</b>
                          <small>
                            {doc.chunks} chunks · {doc.images || 0} images · {formatSize(doc.size)}
                          </small>
                        </span>
                      </label>
                      <button
                        className="ghost danger"
                        onClick={() => deleteDocument(doc.id)}
                        aria-label={`Delete ${doc.name}`}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  );
                })}

                {!visibleDocuments.length && (
                  <div className="empty">
                    <FileText size={22} />
                    <b>{documents.length ? "No match" : "Library is empty"}</b>
                    <span>
                      {documents.length
                        ? "Try a different name."
                        : "Upload PDFs, Word files or text to begin."}
                    </span>
                  </div>
                )}
              </div>

              <button
                className="upload"
                disabled={uploading}
                onClick={() => fileRef.current?.click()}
              >
                {uploading ? (
                  <LoaderCircle className="spin" size={16} />
                ) : (
                  <Upload size={16} />
                )}
                {uploading ? "Reading & indexing…" : "Upload documents"}
              </button>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.txt,.docx"
                hidden
                onChange={(e) => uploadFiles(e.target.files)}
              />
            </>
          ) : (
            <div className="history">
              {conversations.map((c) => (
                <div
                  key={c.id}
                  className={`hist ${c.id === conversationId ? "on" : ""}`}
                >
                  <button
                    className="hist-open"
                    title={c.title}
                    onClick={() => openConversation(c.id)}
                  >
                    <MessageSquare size={14} />
                    <span>{c.title}</span>
                  </button>
                  <button
                    className="ghost danger"
                    aria-label="Delete conversation"
                    onClick={() => deleteConversation(c.id)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
              {!conversations.length && (
                <div className="empty">
                  <MessageSquare size={22} />
                  <b>No chats yet</b>
                  <span>Your conversations will appear here.</span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="side-foot">
          <span className="private">
            <i /> Runs locally · your files never leave this machine
          </span>
        </div>
      </aside>

      {/* ------------------------------- Main -------------------------------- */}
      <main className="stage">
        <header className="bar">
          <button
            className="ghost"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            <PanelLeft size={18} />
          </button>
          <div className="bar-title">
            {conversationId
              ? conversations.find((c) => c.id === conversationId)?.title ||
                "Conversation"
              : "New chat"}
          </div>
          <div className="bar-right">
            <span className={`chip status ${apiReady ? "ok" : ""}`}>
              <i /> {apiReady ? "Online" : "Offline"}
            </span>
            <button
              className="ghost"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
            </button>
          </div>
        </header>

        <section className="scroll" ref={chatRef} onScroll={onChatScroll}>
          {messages.length === 0 ? (
            <div className="hero">
              <div className="orb" aria-hidden="true" />
              <Logo size={52} />
              <h2>
                What would you like
                <br />
                to <span>understand</span> today?
              </h2>
              <p>
                CognifyAI reads your documents and answers with the exact
                passages it used — fully private, running on your own machine.
              </p>
              <div className="cards">
                {SUGGESTIONS.map(({ icon: Icon, title, text }) => (
                  <button
                    key={title}
                    onClick={() => {
                      setQuestion(text);
                      composerRef.current?.focus();
                    }}
                  >
                    <Icon size={18} />
                    <b>{title}</b>
                    <span>{text}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              {messages.map((message, index) =>
                message.role === "user" ? (
                  <div className="turn user" key={index}>
                    {editingIndex === index ? (
                      <div className="edit">
                        <textarea
                          value={editText}
                          autoFocus
                          rows={3}
                          onChange={(e) => setEditText(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !e.shiftKey) {
                              e.preventDefault();
                              ask(editText.trim(), index);
                            }
                            if (e.key === "Escape") setEditingIndex(null);
                          }}
                        />
                        <div className="edit-row">
                          <button onClick={() => setEditingIndex(null)}>
                            <X size={13} /> Cancel
                          </button>
                          <button
                            className="solid"
                            disabled={!editText.trim() || sending}
                            onClick={() => ask(editText.trim(), index)}
                          >
                            <Check size={13} /> Save &amp; resend
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="user-col">
                        <div className="bubble">{message.content}</div>
                        {!sending && (
                          <button
                            type="button"
                            className="edit-btn"
                            onClick={() => startEdit(index)}
                          >
                            <Pencil size={12} /> Edit
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    className={`turn bot ${message.failed ? "failed" : ""}`}
                    key={index}
                  >
                    <div className="bot-mark">
                      <Logo size={26} />
                    </div>
                    <div className="bot-body">
                      {message.failed && (
                        <div className="fail-tag">
                          <AlertCircle size={13} /> Something went wrong
                        </div>
                      )}

                      <div className="prose">
                        {message.streaming && !message.content ? (
                          <span className="typing">
                            <i />
                            <i />
                            <i />
                          </span>
                        ) : (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            urlTransform={(url) => url}
                            components={{
                              a: ({ href, children }) =>
                                href?.startsWith("#src-") ? (
                                  <button
                                    type="button"
                                    className="cite"
                                    title="Show the passage this came from"
                                    onClick={() =>
                                      openEvidence(
                                        index,
                                        Number(href.replace("#src-", ""))
                                      )
                                    }
                                  >
                                    {children}
                                  </button>
                                ) : (
                                  <a href={href} target="_blank" rel="noreferrer">
                                    {children}
                                  </a>
                                ),
                            }}
                          >
                            {linkCitations(message.content)}
                          </ReactMarkdown>
                        )}
                        {message.streaming && message.content && (
                          <span className="caret" />
                        )}
                      </div>

                      {message.images?.length > 0 && (
                        <div className="gallery">
                          <div className="gallery-head">
                            <ImageIcon size={13} /> {message.images.length} image
                            {message.images.length === 1 ? "" : "s"} from your documents
                          </div>
                          <div className="gallery-grid">
                            {message.images.map((img, i) => (
                              <button
                                key={img.id}
                                type="button"
                                className="shot"
                                onClick={() =>
                                  setLightbox({ images: message.images, index: i })
                                }
                              >
                                <img src={`${API}${img.url}`} alt={img.caption || "Document image"} loading="lazy" />
                                <span>
                                  {img.document_name.replace(/\.[^.]+$/, "")}
                                  {img.page ? ` · p.${img.page}` : ""}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {!message.streaming && !message.failed && message.content && (
                        <div className="tools">
                          <button onClick={() => copyMessage(index)}>
                            {copied === index ? (
                              <Check size={13} />
                            ) : (
                              <Copy size={13} />
                            )}
                            {copied === index ? "Copied" : "Copy"}
                          </button>
                          <button
                            disabled={sending}
                            onClick={() => regenerate(index)}
                          >
                            <RefreshCw size={13} /> Retry
                          </button>
                          {message.sources?.length > 0 ? (
                            <button
                              className={`evidence-btn ${
                                evidence?.index === index ? "on" : ""
                              }`}
                              onClick={() =>
                                evidence?.index === index
                                  ? setEvidence(null)
                                  : openEvidence(index)
                              }
                            >
                              <Layers size={13} /> Evidence
                              <em>{message.sources.length}</em>
                            </button>
                          ) : (
                            <span className="mode">General chat</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </section>

        {showJump && messages.length > 0 && (
          <button className="jump" onClick={jumpToBottom} aria-label="Latest">
            <ArrowDown size={16} />
          </button>
        )}

        {(notice || error) && (
          <div className={`toast ${error ? "bad" : ""}`}>
            {error ? <AlertCircle size={15} /> : <CheckCircle2 size={15} />}
            <span>{error || notice}</span>
            <button
              className="ghost"
              onClick={() => {
                setError("");
                setNotice("");
              }}
              aria-label="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
        )}

        <div className="dock">
          <form className="composer" onSubmit={sendQuestion}>
            <textarea
              ref={composerRef}
              value={question}
              rows={1}
              placeholder="Ask CognifyAI about your documents…"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendQuestion();
                }
              }}
            />
            {sending ? (
              <button
                type="button"
                className="go stop"
                onClick={() => abortRef.current?.abort()}
                aria-label="Stop"
              >
                <Square size={14} />
              </button>
            ) : (
              <button
                type="submit"
                className="go"
                disabled={!question.trim()}
                aria-label="Send"
              >
                <ArrowUp size={18} />
              </button>
            )}
          </form>
          <div className="dock-meta">
            <span>
              <Layers size={11} /> {scopeLabel}
            </span>
            <span>Enter to send · Shift+Enter for a new line</span>
          </div>
        </div>
      </main>

      {lightbox && (() => {
        const img = lightbox.images[lightbox.index];
        const move = (d) =>
          setLightbox({
            ...lightbox,
            index: (lightbox.index + d + lightbox.images.length) % lightbox.images.length,
          });
        return (
          <div className="lightbox" onClick={() => setLightbox(null)}>
            <div className="lb-bar" onClick={(e) => e.stopPropagation()}>
              <span>
                {img.document_name}
                {img.page ? ` · page ${img.page}` : ""} · {lightbox.index + 1}/{lightbox.images.length}
              </span>
              <div>
                <a className="ghost" href={`${API}${img.url}`} download aria-label="Download image">
                  <Download size={17} />
                </a>
                <button className="ghost" onClick={() => setLightbox(null)} aria-label="Close">
                  <X size={18} />
                </button>
              </div>
            </div>
            {lightbox.images.length > 1 && (
              <button className="lb-nav left" onClick={(e) => { e.stopPropagation(); move(-1); }} aria-label="Previous">
                <ChevronLeft size={22} />
              </button>
            )}
            <img src={`${API}${img.url}`} alt={img.caption || "Document image"} onClick={(e) => e.stopPropagation()} />
            {lightbox.images.length > 1 && (
              <button className="lb-nav right" onClick={(e) => { e.stopPropagation(); move(1); }} aria-label="Next">
                <ChevronRight size={22} />
              </button>
            )}
            {img.caption && <p className="lb-cap" onClick={(e) => e.stopPropagation()}>{img.caption}</p>}
          </div>
        );
      })()}

      {/* ----------------------------- Evidence ------------------------------ */}
      <aside className="evidence" aria-hidden={!evidence}>
        {evidenceMessage && (
          <>
            <div className="ev-head">
              <div>
                <b>Evidence</b>
                <span>
                  {evidenceMessage.sources.length} passages · {evidenceDocs}{" "}
                  document{evidenceDocs === 1 ? "" : "s"}
                </span>
              </div>
              <button
                className="ghost"
                onClick={() => setEvidence(null)}
                aria-label="Close evidence"
              >
                <X size={17} />
              </button>
            </div>
            <div className="ev-list">
              {evidenceMessage.sources.map((source, i) => (
                <article
                  id={`ev-${i + 1}`}
                  key={`${source.chunk_id}-${i}`}
                  className={`ev ${evidence.focus === i + 1 ? "focus" : ""}`}
                >
                  <header>
                    <span className="ev-n">{i + 1}</span>
                    <div>
                      <b title={source.document_name}>{source.document_name}</b>
                      <small>Chunk #{source.chunk_id}</small>
                    </div>
                    <span className="ev-score">
                      {Math.round(source.score * 100)}%
                    </span>
                  </header>
                  <div className="meter">
                    <i style={{ width: `${Math.round(source.score * 100)}%` }} />
                  </div>
                  <p>{source.content}</p>
                </article>
              ))}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

export default App;
