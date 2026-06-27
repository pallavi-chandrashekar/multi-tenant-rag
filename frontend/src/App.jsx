import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  Send, Upload, Bot, User, Database, Layers,
  Cpu, CheckCircle, AlertCircle, Loader2, Trash2, FolderOpen, FileText, PlusCircle, MessageSquare, Edit, LogOut, Lock
} from 'lucide-react'
import './index.css'

const API_URL = "http://localhost:8000"

// Native browser UUID generator
const generateUUID = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
};

// --- Login / Register gate (shown only when the server has AUTH_ENABLED) ---
function LoginGate({ onLogin, onRegister }) {
  const [mode, setMode] = useState("login")  // "login" | "register"
  const [username, setUsername] = useState("")
  const [password, setPassword] = useState("")
  const [tenant, setTenant] = useState("demo-corp")
  const [role, setRole] = useState("admin")
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setError(null); setBusy(true)
    try {
      if (mode === "register") {
        await onRegister(username, password, tenant, role)
      }
      await onLogin(username, password)
    } catch (err) {
      setError(err.message || "Something went wrong")
    } finally { setBusy(false) }
  }

  return (
    <div className="login-container">
      <form className="login-card" onSubmit={submit}>
        <div className="brand" style={{ justifyContent: 'center', marginBottom: '1rem' }}>
          <Lock size={22} /> <span>Enterprise RAG</span>
        </div>
        <div className="login-title">{mode === "login" ? "Sign in" : "Create an account"}</div>

        <input className="input-field" placeholder="Username" value={username}
          onChange={e => setUsername(e.target.value)} autoFocus />
        <input className="input-field" type="password" placeholder="Password" value={password}
          onChange={e => setPassword(e.target.value)} />

        {mode === "register" && (
          <>
            <input className="input-field" placeholder="Tenant ID" value={tenant}
              onChange={e => setTenant(e.target.value)} />
            <select className="input-field" value={role} onChange={e => setRole(e.target.value)}>
              <option value="viewer">viewer (query, eval)</option>
              <option value="editor">editor (+ ingest)</option>
              <option value="admin">admin (+ delete)</option>
            </select>
          </>
        )}

        {error && <div className="login-error"><AlertCircle size={14} /> {error}</div>}

        <button className="send-btn login-btn" type="submit" disabled={busy}>
          {busy ? "Please wait…" : (mode === "login" ? "Sign in" : "Register & sign in")}
        </button>

        <div className="login-switch">
          {mode === "login" ? (
            <span>Need an account? <a onClick={() => { setMode("register"); setError(null) }}>Register</a></span>
          ) : (
            <span>Have an account? <a onClick={() => { setMode("login"); setError(null) }}>Sign in</a></span>
          )}
        </div>
      </form>
    </div>
  )
}

function App() {
  const [tenantId, setTenantId] = useState("demo-corp")
  const [sessionId, setSessionId] = useState(generateUUID())
  const [chatList, setChatList] = useState([])

  // --- Auth state ---
  const [authEnabled, setAuthEnabled] = useState(false)
  const [authChecked, setAuthChecked] = useState(false)
  const [token, setToken] = useState(() => localStorage.getItem('rag_token') || "")
  const [role, setRole] = useState(() => localStorage.getItem('rag_role') || "")
  const [username, setUsername] = useState(() => localStorage.getItem('rag_user') || "")

  const [documents, setDocuments] = useState([])
  const [query, setQuery] = useState("")
  const [chatHistory, setChatHistory] = useState([{ role: 'ai', content: 'Hello! I am your AI Assistant.' }])
  const [isLoading, setIsLoading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)

  const [contextMenu, setContextMenu] = useState(null)

  const messagesEndRef = useRef(null)
  useEffect(() => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }), [chatHistory])

  // --- Auth helpers ---------------------------------------------------------
  const persistAuth = (tok, r, u, tenant) => {
    setToken(tok); setRole(r); setUsername(u); if (tenant) setTenantId(tenant)
    localStorage.setItem('rag_token', tok)
    localStorage.setItem('rag_role', r)
    localStorage.setItem('rag_user', u)
  }

  const logout = () => {
    setToken(""); setRole(""); setUsername("")
    localStorage.removeItem('rag_token')
    localStorage.removeItem('rag_role')
    localStorage.removeItem('rag_user')
  }

  const login = async (u, p) => {
    const body = new URLSearchParams({ username: u, password: p })
    const res = await fetch(`${API_URL}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body
    })
    if (!res.ok) throw new Error("Invalid username or password")
    const data = await res.json()
    persistAuth(data.access_token, data.role, u, data.tenant_id)
  }

  const register = async (u, p, tenant, r) => {
    const res = await fetch(`${API_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p, tenant_id: tenant, role: r })
    })
    if (!res.ok) {
      const e = await res.json().catch(() => ({}))
      throw new Error(e.detail || "Registration failed")
    }
  }

  // Centralised fetch: attaches the bearer token (and X-Tenant-ID for the
  // header-fallback mode). A 401 forces re-login.
  const apiFetch = async (path, opts = {}) => {
    const headers = { "X-Tenant-ID": tenantId, ...(opts.headers || {}) }
    if (token) headers["Authorization"] = `Bearer ${token}`
    const res = await fetch(`${API_URL}${path}`, { ...opts, headers })
    if (res.status === 401 && authEnabled) logout()
    return res
  }

  // --- On load: discover whether auth is enabled ----------------------------
  useEffect(() => {
    fetch(`${API_URL}/auth/config`)
      .then(r => r.json())
      .then(d => setAuthEnabled(!!d.auth_enabled))
      .catch(() => setAuthEnabled(false))
      .finally(() => setAuthChecked(true))
  }, [])

  // --- Helpers --------------------------------------------------------------
  const handleNewChat = () => {
    setSessionId(generateUUID())
    setChatHistory([{ role: 'ai', content: 'Hello! New session started.' }])
  }

  const loadChat = async (id) => {
    setSessionId(id)
    try {
        const res = await apiFetch(`/api/v1/chats/${id}`)
        const data = await res.json()
        if (data.history) {
            const formatted = data.history.map(m => ({
                role: m.role, content: m.content, strategy_used: m.strategy
            }))
            setChatHistory(formatted)
        }
    } catch (e) { console.error(e) }
  }

  const fetchData = async () => {
    // Don't call protected endpoints until the user has authenticated.
    if (authEnabled && !token) return
    try {
        const docRes = await apiFetch(`/api/v1/documents`);
        const docData = await docRes.json();
        setDocuments(docData.documents || []);

        const chatRes = await apiFetch(`/api/v1/chats`);
        const chatData = await chatRes.json();
        setChatList(chatData || []);

        if (chatData && chatData.length > 0) {
            await loadChat(chatData[0].id)
        } else {
            handleNewChat()
        }
    } catch (e) { console.error(e); }
  }

  useEffect(() => {
    if (!authChecked) return
    fetchData()
  }, [tenantId, token, authChecked])

  // --- Actions --------------------------------------------------------------
  const deleteChat = async (id) => {
    if(!confirm("Are you sure you want to delete this chat?")) return;
    await apiFetch(`/api/v1/chats/${id}`, { method: "DELETE" })

    const remainingChats = chatList.filter(c => c.id !== id)
    setChatList(remainingChats)

    if (sessionId === id) {
        if (remainingChats.length > 0) {
            loadChat(remainingChats[0].id)
        } else {
            handleNewChat()
        }
    }
    setContextMenu(null)
  }

  const renameChat = async (id, currentTitle) => {
    const newTitle = prompt("Enter new chat name:", currentTitle);
    if (!newTitle || !newTitle.trim()) return;

    try {
        const res = await apiFetch(`/api/v1/chats/${id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newTitle })
        });

        if (res.ok) {
            setChatList(prev => prev.map(c => c.id === id ? { ...c, title: newTitle } : c));
        }
    } catch (e) { console.error(e) }
    setContextMenu(null)
  }

  const handleContextMenu = (e, chat) => {
    e.preventDefault()
    setContextMenu({ x: e.clientX, y: e.clientY, chat })
  }

  useEffect(() => {
    const handleClick = () => setContextMenu(null)
    window.addEventListener('click', handleClick)
    return () => window.removeEventListener('click', handleClick)
  }, [])

  const sendMessage = async () => {
    if (!query.trim()) return
    const userMessage = { role: 'user', content: query }
    const newHistory = [...chatHistory, userMessage]
    setChatHistory(newHistory)
    setQuery("")
    setIsLoading(true)

    try {
      const cleanHistory = newHistory.map(m => ({ role: m.role, content: m.content }))
      const response = await apiFetch(`/api/v1/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: userMessage.content,
          tenant_id: tenantId,
          session_id: sessionId,
          chat_history: cleanHistory
        })
      })
      const data = await response.json()

      let botResponse = {
        role: 'ai',
        content: data.answer || "No info found.",
        strategy_used: data.strategy_used || data.mode,
        confidence: data.confidence,
        latency_ms: data.latency_ms,
        sources: data.sources || []
      }

      if (data.results?.length > 0 && !data.answer) {
         botResponse.content = data.results.map((r,i) => `**Src ${i+1}:** ${r.content}`).join('\n\n')
      }

      setChatHistory(prev => [...prev, botResponse])

      const chatRes = await apiFetch(`/api/v1/chats`);
      const chatData = await chatRes.json();
      setChatList(chatData || []);

    } catch (e) { console.error(e) }
    finally { setIsLoading(false) }
  }

  const handleFileUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return
    setUploadStatus({type:'loading', msg:'Uploading'})
    const fd = new FormData(); fd.append("file", files[0])
    await apiFetch(`/api/v1/ingest`, { method: "POST", body: fd })
    setUploadStatus({type:'success', msg:'Done'})

    const docRes = await apiFetch(`/api/v1/documents`);
    const docData = await docRes.json();
    setDocuments(docData.documents || []);
  }

  const handleDeleteDoc = async (id) => {
    if(!confirm("Delete?")) return
    await apiFetch(`/api/v1/documents/${id}`, { method: "DELETE" })
    const docRes = await apiFetch(`/api/v1/documents`);
    const docData = await docRes.json();
    setDocuments(docData.documents || []);
  }

  // --- Auth gate ------------------------------------------------------------
  if (!authChecked) {
    return <div className="login-container"><div className="login-card">Loading…</div></div>
  }
  if (authEnabled && !token) {
    return <LoginGate onLogin={login} onRegister={register} />
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand"><Database size={24}/> <span>Enterprise RAG</span></div>
        <button className="new-chat-btn" onClick={handleNewChat}><PlusCircle size={16}/> New Chat</button>

        <div className="chat-list-section">
            <div className="section-header" style={{marginTop:'1.5rem'}}><MessageSquare size={14}/> Recent Chats</div>
            <div className="chat-history-list">
                {chatList.map(chat => (
                    <div
                        key={chat.id}
                        className={`chat-row ${chat.id === sessionId ? 'active' : ''}`}
                        onClick={() => loadChat(chat.id)}
                        onContextMenu={(e) => handleContextMenu(e, chat)}
                    >
                        {chat.title}
                    </div>
                ))}
            </div>
        </div>

        <div style={{marginTop:'auto'}}>
          {authEnabled ? (
            <div className="control-group">
              <div className="section-header"><User size={14}/> Signed in</div>
              <div className="identity-card">
                <div className="identity-user">{username}</div>
                <div className="identity-meta">
                  <span className="meta-badge">tenant: {tenantId}</span>
                  <span className="meta-badge">{role}</span>
                </div>
                <button className="logout-btn" onClick={logout}><LogOut size={14}/> Sign out</button>
              </div>
            </div>
          ) : (
            <div className="control-group"><div className="section-header"><Layers size={14}/> Tenant ID</div><input className="input-field" value={tenantId} onChange={e => setTenantId(e.target.value)} /></div>
          )}
          <div className="control-group"><label className="upload-zone"><input type="file" hidden onChange={handleFileUpload} />Upload File</label></div>
        </div>
      </aside>

      <main className="main-area">
        <div className="messages-list">
          {chatHistory.map((msg, i) => (
             <div key={i} className="message-item">
                <div className={`avatar ${msg.role}`}>{msg.role==='ai'?<Bot size={20}/>:<User size={20}/>}</div>
                <div className="message-content">
                    {msg.role === 'ai' && (msg.strategy_used || msg.confidence != null) && (
                      <div className="answer-meta">
                        {msg.strategy_used && <span className="meta-badge">Mode: {msg.strategy_used}</span>}
                        {msg.confidence != null && (
                          <span className={`meta-badge confidence ${msg.confidence >= 0.55 ? 'high' : 'low'}`}>
                            Confidence: {(msg.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                        {msg.latency_ms != null && <span className="meta-badge">{msg.latency_ms} ms</span>}
                      </div>
                    )}
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                    {msg.sources?.length > 0 && (
                      <div className="citations">
                        <div className="citations-title"><FileText size={13}/> Sources</div>
                        {msg.sources.map((s, idx) => (
                          <div key={s.chunk_id || idx} className="citation-item">
                            <span className="citation-index">[{idx + 1}]</span>
                            <span className="citation-file">{s.filename}</span>
                            <span className="citation-score">score {(s.retrieval_score ?? s.combined_score ?? 0).toFixed(3)}</span>
                            <div className="citation-snippet">{s.text_snippet}</div>
                          </div>
                        ))}
                      </div>
                    )}
                </div>
             </div>
          ))}
          {isLoading && <div className="message-item"><div className="avatar ai"><Bot size={20}/></div><div className="message-content">Thinking...</div></div>}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-area">
          <div className="input-wrapper"><input value={query} onChange={e=>setQuery(e.target.value)} onKeyDown={e=>e.key==='Enter'&&sendMessage()}/><button className="send-btn" onClick={sendMessage}><Send size={18}/></button></div>
        </div>
      </main>

      <aside className="docs-sidebar">
        <div className="section-header"><FolderOpen size={14}/> Knowledge Base</div>
        {documents.map(d => (
            <div key={d.id} className="doc-item">
                <span className="doc-name">{d.filename}</span>
                <button className="delete-btn" onClick={() => handleDeleteDoc(d.id)}><Trash2 size={16}/></button>
            </div>
        ))}
      </aside>

      {contextMenu && (
        <div className="context-menu" style={{top: contextMenu.y, left: contextMenu.x}}>
            <div onClick={() => renameChat(contextMenu.chat.id, contextMenu.chat.title)} style={{display:'flex', alignItems:'center', gap:5}}>
                <Edit size={14}/> Rename
            </div>
            <div onClick={() => deleteChat(contextMenu.chat.id)} style={{color:'var(--danger)', display:'flex', alignItems:'center', gap:5}}>
                <Trash2 size={14}/> Delete
            </div>
        </div>
      )}
    </div>
  )
}
export default App
