import type { ChatResult } from "../types";

interface Props {
  result: ChatResult | null;
}

function describe(item: ChatResult["items"][number]) {
  return [item.size, item.milk && `${item.milk} milk`, ...item.extras]
    .filter(Boolean)
    .join(" · ");
}

export default function Cart({ result }: Props) {
  const items = result?.items ?? [];

  return (
    <aside className="flex h-full flex-col border-l border-parchment bg-parchment/40">
      <h2 className="border-b border-parchment px-5 py-4 font-medium">Order</h2>

      <div className="flex-1 space-y-3 overflow-y-auto p-5">
        {items.length === 0 && (
          <p className="text-sm text-bark-soft">
            Nothing yet. The order appears here once it is confirmed and checked
            against the menu.
          </p>
        )}

        {items.map((item, i) => (
          <div key={i} className="flex justify-between gap-3 text-sm">
            <div>
              <p className="font-medium">
                {item.quantity > 1 && `${item.quantity} × `}
                {item.name}
              </p>
              {describe(item) && (
                <p className="text-bark-soft">{describe(item)}</p>
              )}
            </div>
            <p className="tabular-nums">{item.line_total.toFixed(2)}</p>
          </div>
        ))}

        {/* Refusals are shown rather than swallowed. The model gets roughly one
            order in twelve wrong, and the server dropping an item silently
            would hide the layer that catches it. */}
        {result?.rejections.map((reason, i) => (
          <p
            key={i}
            className="rounded-lg border border-rust/30 bg-rust/5 px-3 py-2
                       text-sm text-rust"
          >
            Not added — {reason}
          </p>
        ))}
      </div>

      {items.length > 0 && (
        <div className="flex justify-between border-t border-parchment px-5 py-4
                        font-medium">
          <span>Total</span>
          <span className="tabular-nums">{result!.total.toFixed(2)}</span>
        </div>
      )}
    </aside>
  );
}
