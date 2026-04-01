# UniMatrix-Relay (Two-Hop Retrieval)

![UniMatrix-Relay](../assets/arch/umt_relay.png)

**Mechanism.** Use the UniMatrix state to form a first-hop query that retrieves anchor memories, then perform a second-hop refinement through learned relay links before reading out the final value.

**Why it might help.** One-hop retrieval is often brittle when keys collide or when the useful memory is only indirectly related to the final query. Relay adds a small amount of compositional search without paying the full cost of attention over all past tokens.

**Tradeoffs.** More latency and more failure modes: if the first hop is wrong, the second hop usually amplifies the mistake. It also needs stronger diagnostics to stay interpretable.
