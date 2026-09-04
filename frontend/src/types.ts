/**
 * The wire contract: what serving/space/app.py returns, in TypeScript.
 *
 * A cart line is defined by the code that builds it, in
 * serving/space/orders.py. Change it there first, then follow it here.
 */

export interface OrderItem {
  name: string;
  size: string | null;
  milk: string | null;
  extras: string[];
  quantity: number;
  /** Computed by the server from the catalog, never read out of the reply. */
  unit_price: number;
  line_total: number;
}

/** One product on the menu this turn, and why retrieval put it there. */
export interface Chosen {
  name: string;
  why: string;
}

/** One tool the agent called this turn, and what the shop answered. */
export interface Step {
  tool: string;
  arguments: Record<string, unknown> | null;
  result: string;
}

export interface ChatResult {
  /** The reply with the tool call stripped out. */
  text: string;
  items: OrderItem[];
  total: number;
  /** Items the server refused, with the reason. Shown, never swallowed. */
  rejections: string[];
  ordered: boolean;
  voice: string;
  menu: string;
  chosen: Chosen[];
  /** How many times the model spoke before the turn ended. */
  steps: number;
  /** The loop, in order. The agent's working shown. */
  trace: Step[];
}

export interface ShopState {
  voices: string[];
  active: string;
  brand: string;
  shop: string;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}
