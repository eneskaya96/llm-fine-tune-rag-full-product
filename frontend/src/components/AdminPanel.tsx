import { useState } from "react";
import { chat, setVoice } from "../api/client";
import type { ChatResult, ShopState } from "../types";

interface Props {
  state: ShopState;
  onVoiceChange: (state: ShopState) => void;
}

/**
 * The shop's side of the product: pick the brand voice, and see what picking
 * it does. The dropdown is a thin wrapper over `model.set_adapter()` on the
 * Space -- the voices come from the server, so publishing a third adapter
 * reaches this list without a change here.
 */
export default function AdminPanel({ state, onVoiceChange }: Props) {
  const [probe, setProbe] = useState("large flat white with oat pls");
  const [replies, setReplies] = useState<Record<string, ChatResult> | null>(null);
  const [busy, setBusy] = useState(false);

  async function compare() {
    setBusy(true);
    try {
      // A throwaway cart per voice. They are answering the same question at
      // the same time, and one shared thread would have them adding items to
      // each other's order.
      const results = await Promise.all(
        state.voices.map((voice) => chat(probe, [], crypto.randomUUID(), voice)),
      );
      setReplies(Object.fromEntries(state.voices.map((v, i) => [v, results[i]])));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-8">
      <section>
        <h2 className="font-medium">Brand voice</h2>
        <p className="mt-1 text-sm text-bark-soft">
          One base model serves every voice. Choosing here swaps a LoRA adapter
          on the running model — milliseconds, no reload — and changes what
          customers hear.
        </p>
        <div className="mt-4 flex gap-2">
          {state.voices.map((voice) => (
            <button
              key={voice}
              onClick={async () => onVoiceChange(await setVoice(voice))}
              className={
                "rounded-full border px-4 py-2 text-sm capitalize transition " +
                (voice === state.active
                  ? "border-ember bg-ember text-cream"
                  : "border-parchment bg-white hover:border-ember")
              }
            >
              {voice}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h2 className="font-medium">Hear the difference</h2>
        <p className="mt-1 text-sm text-bark-soft">
          The same question to every voice, on one shared prompt that says
          nothing about tone. This does not change the live setting.
        </p>
        <div className="mt-4 flex gap-2">
          <input
            value={probe}
            onChange={(e) => setProbe(e.target.value)}
            className="flex-1 rounded-full border border-parchment bg-white px-4
                       py-2 outline-none focus:border-ember"
          />
          <button
            onClick={compare}
            disabled={busy}
            className="rounded-full bg-bark px-5 py-2 text-sm text-cream
                       transition disabled:opacity-40"
          >
            {busy ? "Asking…" : "Ask all"}
          </button>
        </div>

        {replies && (
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {Object.entries(replies).map(([voice, result]) => (
              <div key={voice} className="rounded-xl bg-white p-4 shadow-sm">
                <p className="text-xs uppercase tracking-wide text-bark-soft">
                  {voice}
                </p>
                <p className="mt-2 text-[15px] leading-relaxed">{result.text}</p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
