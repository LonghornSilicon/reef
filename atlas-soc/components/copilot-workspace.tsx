"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@/components/icons";
import type { AtlasContext, ChatMessage, ChatThread, Citation, ThreadSummary } from "@/lib/domain";

type DisplayMessage = ChatMessage & { imageUrl?: string; pending?: boolean };

function sourceHref(citation: Citation) {
  if (citation.path.startsWith("docs/") && citation.path.endsWith(".md")) {
    return `/wiki/${citation.path.slice(5, -3)}`;
  }
  return `/legacy/${citation.path}`;
}

function MessageBody({ content }: { content: string }) {
  return <>{content.split(/\n\n+/).map((paragraph, index) => <p key={`${index}-${paragraph.slice(0, 12)}`}>{paragraph}</p>)}</>;
}

export function CopilotWorkspace() {
  const params = useSearchParams();
  const context = useMemo<AtlasContext>(() => ({
    blockId: params.get("blockId") || undefined,
    blockName: params.get("blockName") || undefined,
    sourcePath: params.get("sourcePath") || undefined,
    sourceTitle: params.get("sourceTitle") || undefined,
  }), [params]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThread, setActiveThread] = useState<ChatThread | null>(null);
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [threadsOpen, setThreadsOpen] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const threadsToggleRef = useRef<HTMLButtonElement>(null);
  const threadsCloseRef = useRef<HTMLButtonElement>(null);

  async function refreshThreads() {
    const response = await fetch("/api/threads");
    if (response.ok) setThreads((await response.json()).threads);
  }

  useEffect(() => {
    let cancelled = false;
    void fetch("/api/threads")
      .then((response) => response.ok ? response.json() : null)
      .then((data) => { if (!cancelled && data) setThreads(data.threads); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages]);
  useEffect(() => {
    if (!threadsOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeThreads();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [threadsOpen]);

  function openThreads() {
    setThreadsOpen(true);
    requestAnimationFrame(() => threadsCloseRef.current?.focus());
  }

  function closeThreads(restoreFocus = true) {
    setThreadsOpen(false);
    if (restoreFocus) requestAnimationFrame(() => threadsToggleRef.current?.focus());
  }

  async function openThread(id: string) {
    setError("");
    const response = await fetch(`/api/threads/${id}`);
    if (!response.ok) { setError("That conversation is no longer available."); return; }
    const thread = (await response.json()).thread as ChatThread;
    setActiveThread(thread);
    setMessages(thread.messages);
    closeThreads();
  }

  function newThread() {
    setActiveThread(null);
    setMessages([]);
    setError("");
    closeThreads();
  }

  async function removeThread(id: string) {
    const response = await fetch(`/api/threads/${id}`, { method: "DELETE" });
    if (response.ok) {
      if (activeThread?.id === id) newThread();
      await refreshThreads();
    }
  }

  async function sendQuestion() {
    const question = input.trim();
    if (!question || busy) return;
    setBusy(true);
    setError("");
    setInput("");
    const userMessage: DisplayMessage = { id: crypto.randomUUID(), role: "user", content: question, citations: [], createdAt: new Date().toISOString() };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [...current, userMessage, { id: assistantId, role: "assistant", content: "", citations: [], createdAt: new Date().toISOString(), pending: true }]);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, threadId: activeThread?.id, context }),
      });
      if (!response.ok || !response.body) throw new Error((await response.json().catch(() => null))?.error ?? "Atlas could not start this answer.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let threadId = activeThread?.id;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const raw of events) {
          const eventName = raw.match(/^event:\s*(.+)$/m)?.[1];
          const dataLine = raw.match(/^data:\s*(.+)$/m)?.[1];
          if (!dataLine) continue;
          const data = JSON.parse(dataLine);
          if (eventName === "meta") threadId = data.threadId;
          if (eventName === "delta") {
            setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, content: message.content + data.text, pending: true } : message));
          }
          if (eventName === "citations") {
            setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, citations: data.citations } : message));
          }
          if (eventName === "error") throw new Error(data.message);
          if (eventName === "done") {
            setMessages((current) => current.map((message) => message.id === assistantId ? { ...message, pending: false } : message));
          }
        }
      }
      if (threadId) {
        const threadResponse = await fetch(`/api/threads/${threadId}`);
        if (threadResponse.ok) setActiveThread((await threadResponse.json()).thread);
      }
      await refreshThreads();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Atlas could not complete this answer.");
      setMessages((current) => current.filter((message) => message.id !== assistantId));
    } finally {
      setBusy(false);
    }
  }

  async function generateImage() {
    const prompt = input.trim();
    if (!prompt || busy) { setError("Describe the visual explanation you want first."); return; }
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/chat/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, threadId: activeThread?.id, context }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setInput("");
      const createdAt = new Date().toISOString();
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: prompt,
          citations: [],
          createdAt,
        },
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Conceptual visual: ${prompt}`,
          citations: [],
          createdAt,
          imageState: "expired",
          imageUrl: `data:${data.mimeType};base64,${data.base64}`,
        },
      ]);
      if (!activeThread && data.threadId) {
        const threadResponse = await fetch(`/api/threads/${data.threadId}`);
        if (threadResponse.ok) setActiveThread((await threadResponse.json()).thread);
      }
      await refreshThreads();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Atlas could not generate this visual.");
    } finally {
      setBusy(false);
    }
  }

  const contextLabel = context.blockName ?? context.sourceTitle;

  return (
    <div className="copilot-layout">
      <aside className={`thread-ledger${threadsOpen ? " open" : ""}`} id="thread-ledger" aria-label="Conversation history">
        <header>
          <h2>Conversations</h2>
          <span className="thread-header-actions">
            <button type="button" onClick={newThread} aria-label="New conversation"><Icon name="plus" /></button>
            <button ref={threadsCloseRef} className="thread-close" type="button" onClick={() => closeThreads()} aria-label="Close conversations"><Icon name="close" /></button>
          </span>
        </header>
        <div className="thread-list">
          {threads.length ? threads.map((thread) => (
            <div className={activeThread?.id === thread.id ? "thread-row active" : "thread-row"} key={thread.id}>
              <button type="button" onClick={() => void openThread(thread.id)}>
                <strong>{thread.title}</strong>
                <span>{thread.messageCount} entries · expires {new Date(thread.expiresAt).toLocaleDateString()}</span>
              </button>
              <button className="thread-delete" type="button" aria-label={`Delete ${thread.title}`} onClick={() => void removeThread(thread.id)}><Icon name="trash" /></button>
            </div>
          )) : <p className="ledger-empty">Questions you ask will remain here for 30 days.</p>}
        </div>
        <p className="retention-note">Personal threads only · automatic 30-day expiry</p>
      </aside>
      {threadsOpen && <button className="thread-scrim" type="button" aria-label="Close conversations" onClick={() => closeThreads()} />}

      <section className="conversation-desk">
        <header className="conversation-heading">
          <button ref={threadsToggleRef} className="threads-toggle" type="button" onClick={openThreads} aria-expanded={threadsOpen} aria-controls="thread-ledger"><Icon name="menu" />Threads</button>
          <div><h1>{activeThread?.title ?? "Ask from the design record"}</h1><p>Atlas answers from indexed Coral sources and exposes the evidence it used.</p></div>
          <span className="grounded-stamp"><span />Grounded</span>
        </header>
        {contextLabel && (
          <div className="context-strip">
            <Icon name={context.blockId ? "architecture" : "source"} />
            <span><strong>Question context</strong>{contextLabel}</span>
            <Link href={context.blockId ? "/architecture" : `/wiki/${context.sourcePath?.replace(/^docs\//, "").replace(/\.md$/, "")}`}>Inspect</Link>
          </div>
        )}
        <div className="message-scroll" aria-live="polite">
          {!messages.length && (
            <div className="copilot-empty">
              <div className="evidence-orbit" aria-hidden="true"><span>YAML</span><span>MD</span><span>PDF</span><Icon name="copilot" /></div>
              <h2>What do you need to understand?</h2>
              <p>Ask about structure, boot, protocols, instruction flow, ownership, or an unresolved design decision.</p>
              <div className="prompt-starters">
                {["How does DMA reach Coral Core memory?", "What remains undecided in the boot flow?", "Explain the DTCM arbitration path."].map((prompt) => (
                  <button key={prompt} type="button" onClick={() => setInput(prompt)}>{prompt}<Icon name="arrow" /></button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message) => (
            <article className={`message message-${message.role}`} key={message.id}>
              <header><span>{message.role === "assistant" ? "Atlas" : "You"}</span><time>{new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></header>
              <div className="message-copy"><MessageBody content={message.content || (message.pending ? "Looking through the design record…" : "")} /></div>
              {message.imageUrl && (
                <figure className="generated-figure">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={message.imageUrl} alt={message.content} />
                  <figcaption>Conceptual — not authoritative · this image will not be stored</figcaption>
                </figure>
              )}
              {!message.imageUrl && message.imageState === "expired" && <div className="expired-visual"><Icon name="spark" /><span><strong>Ephemeral visual expired</strong>Describe it again to regenerate from current evidence.</span></div>}
              {message.citations.length > 0 && <div className="citation-list"><span>Evidence</span>{message.citations.map((citation, index) => <a href={sourceHref(citation)} key={citation.id} target={citation.path.startsWith("docs/") ? undefined : "_blank"}><b>{index + 1}</b><span><strong>{citation.title}</strong><code>{citation.path}</code></span></a>)}</div>}
            </article>
          ))}
          <div ref={endRef} />
        </div>
        <div className="composer-zone">
          {error && <div className="composer-error" role="alert">{error}<button type="button" onClick={() => setError("")}>Dismiss</button></div>}
          <div className="composer">
            <textarea value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendQuestion(); } }} placeholder="Ask a question about Coral…" rows={2} disabled={busy} />
            <div className="composer-actions">
              <button type="button" className="image-action" onClick={() => void generateImage()} disabled={busy || !input.trim()}><Icon name="spark" />Visualize</button>
              <span>Enter to send · Shift+Enter for a new line</span>
              <button type="button" className="send-action" onClick={() => void sendQuestion()} disabled={busy || !input.trim()} aria-label="Send question"><Icon name="send" /></button>
            </div>
          </div>
          <p>Atlas can be wrong. Verify design decisions in the cited source before acting.</p>
        </div>
      </section>
    </div>
  );
}
