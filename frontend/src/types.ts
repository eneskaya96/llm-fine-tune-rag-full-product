/**
 * The wire contract, mirroring shared/order_schema.json.
 *
 * That file is the single definition of what an order is: the training data
 * emits it, the evaluator scores against it, and serving/space/orders.py
 * validates against it. This is the same shape in TypeScript. Change the JSON
 * schema first, then follow it here.
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
