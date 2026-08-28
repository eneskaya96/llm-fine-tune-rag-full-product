import { useEffect, useRef, useState } from "react";
import type { Turn } from "../types";

interface Props {
  turns: Turn[];
  pending: boolean;
  error: string | null;
  onSend: (message: string) => void;
}

const OPENERS = [
  "large flat white with oat pls",
  "something cold and refreshing",
  "can I get an iced latte",
];

export default function Chat({ turns, pending, error, onSend }: Props) {
  const [draft, setDraft] = useState("");
  const tail = useRef<HTMLDivElement>(null);

  useEffect(() => {
    tail.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, pending]);

  function send(message: string) {
    if (!message.trim() || pending) return;
    setDraft("");
    onSend(message.trim());
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-5">
        {turns.length === 0 && (
          <div className="pt-10 text-center">
            <p className="text-bark-soft">What can we get you?</p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {OPENERS.map((o) => (
                <button
                  key={o}
                  onClick={() => send(o)}
                  className="rounded-full border border-parchment bg-white px-3 py-1.5
                             text-sm text-bark-soft transition hover:border-ember
                             hover:text-ember"
                >
                  {o}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <div
            key={i}
            className={turn.role === "user" ? "flex justify-end" : "flex justify-start"}
          >
            <p
              className={
                "max-w-[80%] rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed " +
                (turn.role === "user"
                  ? "bg-bark text-cream"
                  : "bg-white text-bark shadow-sm")
              }
            >
              {turn.content}
            </p>
          </div>
        ))}

        {pending && (
          <p className="text-sm text-bark-soft">
            Thinking…{" "}
            <span className="opacity-70">
              the first reply can take a minute while the GPU wakes up
            </span>
          </p>
        )}
        {error && <p className="text-sm text-rust">{error}</p>}
        <div ref={tail} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(draft);
        }}
        className="flex gap-2 border-t border-parchment p-4"
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="a flat white, please"
          className="flex-1 rounded-full border border-parchment bg-white px-4 py-2.5
                     outline-none focus:border-ember"
        />
        <button
          disabled={pending || !draft.trim()}
          className="rounded-full bg-ember px-5 py-2.5 font-medium text-cream
                     transition disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}
