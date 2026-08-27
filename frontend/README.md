# Frontend

Not built yet.

React app on Vercel, talking to the HF Space over HTTP.

## Planned

```
src/App.tsx
src/components/Chat.tsx        customer conversation
src/components/Cart.tsx        order as it builds
src/components/AdminPanel.tsx  brand voice selector
src/api/client.ts              calls into serving/
```

## Why split from serving

Keeping the product layer separate from the ML service is how this would be
built for real, and it is the only free arrangement that also gets a GPU: the
Space runs on ZeroGPU (Gradio SDK required), while React deploys to Vercel.

The admin panel's voice selector is a thin wrapper over the adapter swap — one
dropdown, one `set_adapter()` call on the other side.

Order rendering follows `shared/order_schema.json`, the same contract the model
emits and the serving layer executes.
