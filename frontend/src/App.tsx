import { useEffect, useState } from "react";
import AdminPanel from "./components/AdminPanel";
import Cart from "./components/Cart";
import Chat from "./components/Chat";
import { chat, getState } from "./api/client";
import type { ChatResult, ShopState, Turn } from "./types";

type Tab = "order" | "admin";

/**
 * Names this browser's cart on the server.
 *
 * Made once per load rather than stored: a reload starts a new conversation
 * anyway, since the transcript lives in state here, and a thread whose
 * transcript is gone would have the server remembering a cart nobody can see.
 */
const THREAD = crypto.randomUUID();

export default function App() {
  const [tab, setTab] = useState<Tab>("order");
  const [shop, setShop] = useState<ShopState | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [result, setResult] = useState<ChatResult | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getState()
      .then(setShop)
      .catch(() => setError("Could not reach the shop. The Space may be asleep."));
  }, []);

  async function send(message: string) {
    // The history sent is the conversation *before* this turn; the server
    // appends the message itself. Retrieval reads all of it, which is what
    // keeps a product ordered three turns ago on the menu.
    const history = turns;
    setTurns([...history, { role: "user", content: message }]);
    setPending(true);
    setError(null);
    try {
      const reply = await chat(message, history, THREAD);
      setTurns((t) => [...t, { role: "assistant", content: reply.text }]);
      setResult(reply);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }

  if (!shop) {
    return (
      <div className="grid h-full place-items-center text-bark-soft">
        {error ?? "Opening up…"}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-parchment
                         px-6 py-4">
        <div>
          <h1 className="text-lg font-medium">{shop.brand}</h1>
          <p className="text-xs text-bark-soft">
            speaking in its <span className="text-ember">{shop.active}</span> voice
          </p>
        </div>
        <nav className="flex gap-1 rounded-full bg-parchment p-1 text-sm">
          {(["order", "admin"] as Tab[]).map((name) => (
            <button
              key={name}
              onClick={() => setTab(name)}
              className={
                "rounded-full px-4 py-1.5 capitalize transition " +
                (tab === name ? "bg-white shadow-sm" : "text-bark-soft")
              }
            >
              {name}
            </button>
          ))}
        </nav>
      </header>

      {tab === "admin" ? (
        <div className="flex-1 overflow-y-auto">
          <AdminPanel state={shop} onVoiceChange={setShop} />
        </div>
      ) : (
        <div className="grid flex-1 grid-cols-1 overflow-hidden sm:grid-cols-[1fr_320px]">
          <div className="flex flex-col overflow-hidden">
            <div className="flex-1 overflow-hidden">
              <Chat turns={turns} pending={pending} error={error} onSend={send} />
            </div>
            {result && (
              /* The retrieved menu, shown on purpose. The shop has 28 products
                 and the model was trained on menus of 4-9 drinks, so something
                 chose these -- and a demo that hides the choosing looks like
                 magic instead of like retrieval. */
              <details className="border-t border-parchment px-5 py-3 text-sm">
                <summary className="cursor-pointer text-bark-soft">
                  {result.chosen.length} of 28 products were put in front of the
                  model, and it spoke {result.steps} time
                  {result.steps === 1 ? "" : "s"}
                </summary>
                <ul className="mt-3 space-y-1">
                  {result.chosen.map((product) => (
                    <li key={product.name} className="flex justify-between gap-4">
                      <span>{product.name}</span>
                      <span className="text-bark-soft">{product.why}</span>
                    </li>
                  ))}
                </ul>
                {result.trace.length > 0 && (
                  /* The loop is the claim this version makes, so it is on the
                     page: the model called these, read what came back, and
                     answered that. */
                  <ol className="mt-4 space-y-1 border-t border-parchment pt-3">
                    {result.trace.map((step, index) => (
                      <li key={index} className="flex justify-between gap-4">
                        <span className="font-mono text-xs text-ember">
                          {step.tool}
                        </span>
                        <span className="text-right text-bark-soft">
                          {step.result}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
              </details>
            )}
          </div>
          <Cart result={result} />
        </div>
      )}
    </div>
  );
}
