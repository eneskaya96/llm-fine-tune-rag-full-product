# Frontend

React on Vercel, talking to the Hugging Face Space over HTTP.

```
src/App.tsx                 two screens, and the conversation state
src/api/client.ts           the only file that knows about Gradio
src/types.ts                mirrors shared/order_schema.json
src/components/Chat.tsx     customer conversation
src/components/Cart.tsx     the order as it is validated
src/components/AdminPanel.tsx  brand voice selector
```

Vite + React + TypeScript + Tailwind v4. No router and no state library: two
screens and one conversation do not need either, and both are a local change
to `App.tsx` when they do.

## Running it

```bash
npm install
npm run dev
```

It talks to the live Space by default, so no GPU is needed locally. Point it
somewhere else with `.env`:

```
VITE_SPACE_ID=eneskaya96/coffee-order-voice-swap   # a different shop
VITE_API_BASE=http://127.0.0.1:7860/gradio_api/call  # a Space running locally
```

## Why there is no Gradio client library

Gradio's REST protocol is two steps: POST the arguments and get an event id,
then GET that id as an event stream and read the completion. That is about
thirty lines, which is less than the dependency costs, and it keeps the
protocol readable in one place. `api/client.ts` is that place — moving to a
different backend means rewriting it and nothing else.

## Why the frontend is split from serving

Keeping the product layer separate from the ML service is how this would be
built for real, and it is the only free arrangement that also gets a GPU: the
Space needs the Gradio SDK to reach ZeroGPU, while React deploys to Vercel.

## Two things the UI shows on purpose

**Refused items.** The adapters are wrong about roughly one order in twelve, so
`serving/space/orders.py` checks every item against the menu that was actually
retrieved and drops what fails. The cart shows the drop and the reason. Hiding
it would hide the layer that makes the model's output safe to act on.

**What retrieval chose.** The shop has 28 products and the model was trained on
menus of 4–9 drinks, so something picks. The footer under the conversation
lists what went in and whether each came from the query or from something named
earlier. A demo that hides the choosing looks like magic rather than retrieval.

## Deploying

Vercel, from this repository, with the root directory set to `frontend`. Set
`VITE_SPACE_ID` in the project's environment variables if it is not the default
shop. The first request after a quiet period wakes the Space and can take a
minute; the chat says so rather than looking stuck.
