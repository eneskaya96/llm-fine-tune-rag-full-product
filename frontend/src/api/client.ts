/**
 * The only file that knows the backend is a Gradio Space.
 *
 * Gradio's REST protocol is two steps: POST the arguments and get an event id,
 * then GET that id as a server-sent event stream and read the completion. It
 * is short enough to write out, which is why there is no client library here
 * -- one less dependency, and the protocol stays legible.
 *
 * Everything above this file speaks in the types from ../types. Moving to a
 * different backend means rewriting this file and nothing else.
 */

import type { ChatResult, ShopState, Turn } from "../types";

const SPACE = import.meta.env.VITE_SPACE_ID ?? "eneskaya96/coffee-order-voice-swap";

/**
 * VITE_API_BASE points at a Gradio instance directly and wins when set -- a
 * Space running locally, or a self-hosted one. Otherwise the Space id is
 * turned into its hf.space address.
 */
const BASE =
  import.meta.env.VITE_API_BASE ??
  `https://${SPACE.replace("/", "-").toLowerCase()}.hf.space/gradio_api/call`;

async function call<T>(endpoint: string, data: unknown[]): Promise<T> {
  const started = await fetch(`${BASE}/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  if (!started.ok) {
    throw new Error(`${endpoint} refused the request (${started.status})`);
  }
  const { event_id: eventId } = (await started.json()) as { event_id: string };

  const stream = await fetch(`${BASE}/${endpoint}/${eventId}`);
  return readCompletion<T>(await stream.text(), endpoint);
}

/**
 * Pull the payload out of the event stream.
 *
 * The body arrives as `event: <name>` / `data: <json>` pairs. Only `complete`
 * carries the return value; `error` carries the reason and has to be raised
 * rather than parsed, or a failure would look like an empty reply.
 */
function readCompletion<T>(body: string, endpoint: string): T {
  let event = "";
  for (const line of body.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:") && event === "complete") {
      return (JSON.parse(line.slice(5)) as T[])[0];
    } else if (line.startsWith("data:") && event === "error") {
      throw new Error(`${endpoint} failed: ${line.slice(5).trim()}`);
    }
  }
  throw new Error(`${endpoint} returned no result`);
}

export const getState = () => call<ShopState>("get_state", []);

export const setVoice = (voice: string) =>
  call<ShopState>("set_voice", [voice]);

/**
 * One customer turn.
 *
 * `thread` names the cart. The transcript is ours to draw and travels in
 * `history`; the cart is the server's, because a cart this side could edit
 * would undo the checking that is the point of the order layer. So the turn
 * sends what was said and a name for what has been ordered.
 *
 * `voice` is optional and overrides the shop setting for this request only --
 * that is what lets the admin screen ask the same question in two voices
 * without changing what customers hear. It goes over the wire as "" rather
 * than null when unset: the endpoint declares a plain `str`, and how strictly
 * a given gradio version reads that is not worth finding out in production.
 */
export const chat = (
  message: string,
  history: Turn[],
  thread: string,
  voice?: string,
) => call<ChatResult>("chat", [message, history, voice ?? "", thread]);
