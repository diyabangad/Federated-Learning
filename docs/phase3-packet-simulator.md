# Phase 3: Deterministic Packet Simulator

## Packet and Time Units

`Packet` records an ID, satellite source, destination node, UTC generation time, packet size in bits, current node, hops, per-component delay in seconds, status, drop reason, and route. `PacketFlow.generation_times_seconds` are offsets from the first topology sample. The simulator horizon, deadline, processing delay, contact duration, and event times are seconds; energy is joules; packet size is bits; route hop count is integer link traversals.

## Event Semantics

`PacketSimulator` processes stable, timestamp-ordered events:

1. `packet_generated` creates the packet at a configured fixed time.
2. `route_selected` chooses the next hop using hop, delay, energy-constrained, or contact-aware routing.
3. If no usable link exists yet, the packet waits until the next contact-plan sample. If a packet competes for a satellite's half-duplex radio, it waits until that radio is available.
4. Processing occurs first if configured. TX/RX energy affordability is checked before transmission.
5. `transmission_started` debits sender TX energy once. `transmission_completed` occurs after processing plus packet transmission time.
6. Propagation continues after transmission completion. `packet_arrived` occurs after propagation delay and debits satellite receiver energy once.
7. The packet is delivered, routed onward, or dropped for deadline, energy, route, contact, hop-limit, or horizon reasons.

The current radio policy reserves the transmitter and receiver radios until packet arrival (half-duplex, single-radio endpoint model). Concurrent attempts are serialized in stable event insertion order; there is no finite buffer capacity, MAC backoff, stochastic loss, or link-layer retransmission. A route rejected before TX incurs no communication-energy debit. Compute energy incurred before the affordability decision remains charged as a modeled operation.

For a configured edge, packet delay components default to propagation `range_km / c`, transmission `packet_bits / transmission_rate_bps`, and optional `processing_s`; an edge may explicitly provide `propagation_s`, `transmission_s`, and `processing_s`. Contact must cover processing plus transmission before TX starts. Once TX completes, propagation is allowed to finish even if the contact subsequently closes. Route selection is reevaluated at packet arrival using the graph sample active at that time.

## Energy Accounting

For a packet of $L$ bits over an edge of distance $d$ km:

- TX energy: $L(E_{tx,electronics} + \epsilon d^2)$ J, debited at transmission start.
- RX energy: $L E_{rx,electronics}$ J, debited at arrival only for a satellite receiver.
- Compute energy: configured joules per forwarding hop, charged before transmission.
- Residual satellite energy: initial energy minus accumulated satellite communication and compute energy.

Both endpoint budgets are checked before TX begins, so rejected transmissions do not partially charge TX or RX. Ground-station energy is outside the satellite ledger. FL energy remains separate from this packet simulator. Coefficients are provisional inputs and are not hardware-calibrated.

## Metrics

`PacketSimulationResult` reports PDR (`None` for zero generated packets), average end-to-end and component delays over delivered packets (`None` for zero deliveries), average delivered-packet hop count (`None` for zero deliveries), total satellite communication/compute energy, per-satellite energy ledgers, residual energy, and drop counts by reason.

## Tests and Limitations

`tests/test_packet_sim.py` contains ten deterministic small-graph tests for separated transmission/arrival events, processing order, future-contact waits, too-short contacts, same-radio contention, TX/RX energy charging, exact-budget sends, energy-constrained relays, denominator-zero metrics, and reproducibility.

This is a deterministic prototype, not a complete network simulator. It models one half-duplex radio per endpoint and serializes concurrent traffic; it does not model finite queues, queue overflow, MAC scheduling, channel errors, packet fragmentation, or dynamic compute contention. Time-dependent contacts are sampled; the simulator checks transmission fit using stored remaining-contact metadata and does not interpolate a live orbital crossing during packet execution. These assumptions must be held constant in comparisons and reported with results.
