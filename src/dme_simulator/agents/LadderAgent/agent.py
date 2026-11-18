"""LadderAgent: lestvica cena + prilagodljivi dodatak na aukciji.

Kratak pregled (detaljno objasnjenje je u razgovoru / README tima):

1. CENE    - svaki proizvod ima referentnu cenu R. Roba se nudi na nekoliko
             nivoa oko R ("lestvica"). Posle svakog poteza R se pomera gore ako
             se skupa roba prodaje, a dole ako roba stoji.
             Na pocetku je "faza otkrivanja": nivoi su R, 2R, 4R, 8R pa se brzo
             nadje red velicine cene koju kupci placaju.
2. KOLICINA- Q je procena koliko komada proizvoda prodajemo po potezu. Proizvodimo
             tako da na lageru imamo oko Q * STOCK_BUFFER komada.
3. AUKCIJA - za svaki materijal nudimo rezervna_cena * m. Dodatak m raste kad
             izgubimo (delimicno ili potpuno), a polako pada kad sve dobijemo.
             m je ograniceno vrednoscu materijala u proizvodu.

Sve se izvodi iz recepata i onoga sto agent vidi; nista nije vezano za
konkretan scenario.
"""

from __future__ import annotations

from math import ceil

from dme_simulator.agents import (
    AssemblyOrder,
    BaseAgent,
    Bid,
    Decision,
    Listing,
    Observation,
)
from dme_simulator.config import GoodsConfig


class LadderAgent(BaseAgent):
    # ------------------------------------------------------------------
    # PARAMETRI - ovde se strategija najlakse menja
    # ------------------------------------------------------------------

    # --- faza otkrivanja cene ---
    DISCOVERY_START_MULT = 4.0      # pocetni R = trosak materijala * ovo
    DISCOVERY_MIN_PRICE = 10.0      # ... ali ne manje od ovoga
    DISCOVERY_TIERS = (1.0, 2.0, 4.0, 8.0)  # nivoi u fazi otkrivanja (puta R)
    DISCOVERY_NOTHING_SOLD = 0.6    # nista nije prodato -> R * ovo

    # --- normalna lestvica ---
    TIER_LOW = 0.90                 # nizak nivo (puta R)
    TIER_MID = 1.0                  # srednji nivo
    TIER_HIGH = 1.12                # visok nivo
    TIER_PROBE = 1.35               # sonda (1 komad)
    SHARE_LOW = 0.30                # udeo robe na niskom nivou
    SHARE_HIGH = 0.20               # udeo robe na visokom nivou
    OVERSTOCK_LOW_MULT = 0.75       # kad ima previse robe, nizak nivo ide ovde
    OVERSTOCK_LOW_SHARE = 0.5

    # --- pomeranje R posle poteza ---
    UP_PROBE_SOLD = 1.10            # prodata sonda -> R * ovo
    UP_HIGH_SOLD = 1.05             # visok nivo prodat bar napola
    UP_HIGH_SOME = 1.02             # visok nivo prodat bar 1 komad
    UP_ALL_SOLD = 1.04              # sve rasprodato (malo robe, bez visokog nivoa)
    DOWN_PARTIAL = 0.93             # prodato manje od pola
    DOWN_NOTHING = 0.85             # nista prodato
    MIN_MARGIN = 1.10               # R nikad ispod troska * ovo
    LISTING_FLOOR = 1.05            # nijedna cena ispod troska * ovo

    # --- kolicine ---
    INITIAL_Q = 2.0                 # pocetna procena prodaje po potezu
    DISCOVERY_STOCK = 4             # koliko komada drzimo dok otkrivamo cenu
    Q_SELLOUT_GROWTH = 1.25         # rasprodato -> Q raste
    Q_SMOOTHING = 0.4               # tezina nove prodaje u proseku
    MIN_Q = 1.0
    STOCK_BUFFER = 1.7              # ciljni lager = Q * ovo

    # --- aukcija ---
    M_START = 1.05                  # pocetni dodatak na rezervnu cenu
    M_MIN = 1.01
    M_DOWN = 0.93                   # dobili sve -> m * ovo
    M_UP_PARTIAL = 1.30             # dobili deo -> m * ovo
    M_UP_NONE = 1.50                # nismo dobili nista -> m * ovo
    VALUE_CAP_SHARE = 0.8           # najvise ovoliki deo vrednosti materijala
    BUDGET_SHARE = 0.6              # najvise ovoliki deo kapitala po aukciji

    # --- pracenje konkurenata ---
    # Testirano protiv stare verzije (20 seed-ova, 6 scenarija): nijedna
    # funkcija nije donela merljiv pomak. (a) je neutralna i ostaje ukljucena;
    # (b) neutralna; (c) i (d) su blago odmagale, pa su iskljucene.
    # (a) tudje prodaje, ako ih scenario prikazuje (observe_all_transactions)
    USE_VISIBLE_SALES = True
    # (b) tudje cene na aukciji, ako ih scenario prikazuje
    USE_VISIBLE_BIDS = False
    VISIBLE_BID_STEP = 1.01         # nudi ovoliko iznad najvise tudje cene
    # (c) javne zalihe gotovih proizvoda konkurenata
    USE_COMP_STOCK = False
    COMP_STOCK_RATIO = 4.0          # konkurenti imaju > ovo * Q komada ...
    COMP_STOCK_BUFFER = 1.2         # ... onda drzimo manji lager (Q * ovo)
    # (d) javne zalihe sirovina konkurenata (ko gomila materijal)
    USE_COMP_RAW = False
    COMP_RAW_PRESSURE = 0.3         # konkurenti uzeli > ovaj deo lota ...
    COMP_RAW_BUMP = 1.10            # ... onda unapred dizemo ponudu

    def __init__(self) -> None:
        self.my_id: str | None = None
        self.R: dict[str, float] = {}          # referentna cena po proizvodu
        self.discovery: dict[str, bool] = {}   # da li je proizvod u otkrivanju
        self.Q: dict[str, float] = {}          # procena prodaje po potezu
        self.m: dict[str, float] = {}          # dodatak na aukciji po materijalu
        self.last_reserve: dict[str, float] = {}
        self.last_listings: list[tuple[str, float, int]] = []
        self.last_bids: dict[str, int] = {}
        self.sales_seen = 0
        self.auctions_seen = 0
        # konkurenti
        self.prev_public: dict[str, dict[str, int]] = {}
        self.comp_finished: dict[str, int] = {}     # njihove zalihe proizvoda
        self.comp_raw_gain: dict[str, int] = {}     # koliko su sirovina dodali
        self.foreign_sales: dict[str, list[float]] = {}  # tudje prodajne cene
        self.foreign_bid_max: dict[str, float] = {}  # najvisa tudja cena na aukciji
        self.fill_short: dict[str, bool] = {}       # da li nismo dobili sve

    # ------------------------------------------------------------------
    # glavna funkcija
    # ------------------------------------------------------------------

    def decide(self, state: Observation, goods_config: GoodsConfig) -> Decision:
        self._identify(state)

        for lot in state.auction_lots:
            self.last_reserve[lot.good] = lot.reserve_price
        lots = {lot.good: lot for lot in state.auction_lots}

        recipe_for = self._cheapest_recipes(goods_config)
        cost = {
            good: self._recipe_cost(recipe)
            for good, (_, recipe) in recipe_for.items()
        }

        for good in goods_config.finished_goods:
            if good not in self.R:
                base = cost.get(good, 0.0) * self.DISCOVERY_START_MULT
                self.R[good] = max(base, self.DISCOVERY_MIN_PRICE)
                self.discovery[good] = True
                self.Q[good] = self.INITIAL_Q

        # 1) ucimo iz proslog poteza
        self._observe_competitors(state, goods_config)
        self._learn_from_sales(state, cost)
        self._learn_from_auction(state)

        # 2) plan proizvodnje
        production = self._plan_production(state, goods_config, recipe_for, cost)

        # 3) ponude na aukciji
        bids = self._make_bids(state, lots, recipe_for, cost, production)

        # 4) redosled proizvodnje: najveca marza prvo
        assembly = []
        for good in sorted(
            production, key=lambda g: self.R[g] - cost.get(g, 0.0), reverse=True
        ):
            if production[good] > 0:
                assembly.append(AssemblyOrder(recipe_for[good][0], production[good]))

        # 5) lestvica cena
        listings = self._make_listings(state, goods_config, cost, production)

        self.last_listings = [(l.good, l.price, l.quantity) for l in listings]
        self.last_bids = {b.good: b.quantity for b in bids}
        return Decision(bids=bids, assembly=assembly, listings=listings)

    # ------------------------------------------------------------------
    # pomocne funkcije
    # ------------------------------------------------------------------

    def _identify(self, state: Observation) -> None:
        """Nadji svoj team ID poredjenjem javnog kapitala i zaliha."""
        if self.my_id is not None:
            return
        candidates = [
            team
            for team, capital in state.public_capital.items()
            if abs(capital - state.capital) < 1e-6
            and state.public_inventories.get(team, {}) == state.inventory
        ]
        if len(candidates) == 1:
            self.my_id = candidates[0]

    def _is_mine(self, team: str) -> bool:
        return self.my_id is None or team == self.my_id

    def _recipe_cost(self, recipe) -> float:
        return sum(
            self.last_reserve.get(material, 0.0) * amount
            for material, amount in recipe.inputs.items()
        )

    def _cheapest_recipes(self, goods_config: GoodsConfig):
        best: dict[str, tuple[str, object]] = {}
        for name, recipe in goods_config.recipes.items():
            good = recipe.output_good
            if good not in best or self._recipe_cost(recipe) < self._recipe_cost(
                best[good][1]
            ):
                best[good] = (name, recipe)
        return best

    # --- ucenje -------------------------------------------------------

    def _observe_competitors(self, state: Observation, goods_config: GoodsConfig) -> None:
        """Iz javnih zaliha: koliko robe drze konkurenti i koliko sirovina dodaju."""
        self.comp_finished = {}
        self.comp_raw_gain = {}
        if self.my_id is None:
            return
        for team, inventory in state.public_inventories.items():
            if team == self.my_id:
                continue
            for good in goods_config.finished_goods:
                self.comp_finished[good] = (
                    self.comp_finished.get(good, 0) + inventory.get(good, 0)
                )
            previous = self.prev_public.get(team)
            if previous is not None:
                for material in goods_config.raw_materials:
                    gain = inventory.get(material, 0) - previous.get(material, 0)
                    if gain > 0:
                        self.comp_raw_gain[material] = (
                            self.comp_raw_gain.get(material, 0) + gain
                        )
        self.prev_public = {
            team: dict(inventory) for team, inventory in state.public_inventories.items()
        }

    def _learn_from_sales(self, state: Observation, cost: dict[str, float]) -> None:
        new_sales = state.sales_transactions[self.sales_seen:]
        self.sales_seen = len(state.sales_transactions)

        sold_at: dict[tuple[str, float], int] = {}
        self.foreign_sales = {}
        for sale in new_sales:
            if self._is_mine(sale.seller):
                key = (sale.good, round(sale.price, 6))
                sold_at[key] = sold_at.get(key, 0) + sale.quantity
            else:
                self.foreign_sales.setdefault(sale.good, []).extend(
                    [sale.price] * sale.quantity
                )

        tiers_by_good: dict[str, list[tuple[float, int, int]]] = {}
        for good, price, qty in self.last_listings:
            key = (good, round(price, 6))
            sold = min(qty, sold_at.get(key, 0))
            sold_at[key] = sold_at.get(key, 0) - sold
            tiers_by_good.setdefault(good, []).append((price, qty, sold))

        for good, tiers in tiers_by_good.items():
            listed = sum(q for _, q, _ in tiers)
            sold = sum(s for _, _, s in tiers)
            if listed <= 0:
                continue
            top_price, _, top_sold = max(tiers, key=lambda t: t[0])
            frac = sold / listed
            price_before = self.R[good]

            if self.discovery[good]:
                if sold == 0:
                    self.R[good] *= self.DISCOVERY_NOTHING_SOLD
                else:
                    best_sold = max(p for p, _, s in tiers if s > 0)
                    self.R[good] = best_sold
                    if top_sold == 0:
                        self.discovery[good] = False
            else:
                # prodaja na visokom nivou (iznad R, ali ispod sonde)
                high = [(q, s) for p, q, s in tiers
                        if price_before * 1.0001 < p < price_before * self.TIER_PROBE * 0.999]
                high_q = sum(q for q, _ in high)
                high_s = sum(s for _, s in high)
                if top_sold > 0 and top_price >= price_before * self.TIER_PROBE * 0.999:
                    self.R[good] *= self.UP_PROBE_SOLD
                elif high_q > 0 and high_s >= 0.5 * high_q:
                    self.R[good] *= self.UP_HIGH_SOLD
                elif high_s > 0:
                    self.R[good] *= self.UP_HIGH_SOME
                elif frac >= 0.999:
                    self.R[good] *= self.UP_ALL_SOLD
                elif frac >= 0.5:
                    pass
                elif frac > 0:
                    self.R[good] *= self.DOWN_PARTIAL
                else:
                    self.R[good] *= self.DOWN_NOTHING

            self.R[good] = max(self.R[good], cost.get(good, 0.0) * self.MIN_MARGIN)

            # procena prodaje po potezu: ako se rasprodalo sve sto je bilo
            # po ceni <= R (nizak i srednji nivo), kupaca ima vise nego robe.
            core = [(q, s) for p, q, s in tiers if p <= price_before * self.TIER_MID + 1e-6]
            core_sold_out = bool(core) and all(s >= q for q, s in core)
            if frac >= 0.999 or core_sold_out:
                self.Q[good] = max(self.Q[good], sold) * self.Q_SELLOUT_GROWTH
            else:
                self.Q[good] = (
                    (1 - self.Q_SMOOTHING) * self.Q[good] + self.Q_SMOOTHING * sold
                )
            self.Q[good] = max(self.MIN_Q, self.Q[good])

        # (a) tudje prodaje (vidljive samo u nekim scenarijima): ako drugi
        # prodaju skuplje od nas, nasa cena je verovatno preniska.
        if self.USE_VISIBLE_SALES:
            for good, prices in self.foreign_sales.items():
                if good not in self.R or not prices:
                    continue
                prices.sort()
                median = prices[len(prices) // 2]
                if self.discovery[good]:
                    self.R[good] = max(median, cost.get(good, 0.0) * self.MIN_MARGIN)
                    self.discovery[good] = False
                elif median > self.R[good] * self.TIER_HIGH:
                    self.R[good] *= self.UP_HIGH_SOLD

    def _learn_from_auction(self, state: Observation) -> None:
        new = state.auction_transactions[self.auctions_seen:]
        self.auctions_seen = len(state.auction_transactions)
        won: dict[str, int] = {}
        self.foreign_bid_max = {}
        for result in new:
            if self._is_mine(result.buyer):
                won[result.good] = won.get(result.good, 0) + result.quantity
            else:
                self.foreign_bid_max[result.good] = max(
                    self.foreign_bid_max.get(result.good, 0.0), result.price
                )
        self.fill_short = {
            good: won.get(good, 0) < requested
            for good, requested in self.last_bids.items()
            if requested > 0
        }

        for good, requested in self.last_bids.items():
            if requested <= 0:
                continue
            m = self.m.get(good, self.M_START)
            got = won.get(good, 0)
            if got >= requested:
                m *= self.M_DOWN
            elif got > 0:
                m *= self.M_UP_PARTIAL
            else:
                m *= self.M_UP_NONE
            self.m[good] = max(self.M_MIN, m)

    # --- odluke -------------------------------------------------------

    def _plan_production(self, state, goods_config, recipe_for, cost):
        production: dict[str, int] = {}
        for good in goods_config.finished_goods:
            if good not in recipe_for:
                continue
            stock = state.inventory.get(good, 0)
            if self.discovery[good]:
                target = max(self.DISCOVERY_STOCK, ceil(self.Q[good]))
            else:
                buffer = self.STOCK_BUFFER
                # (c) konkurenti vec drze mnogo ovog proizvoda -> ocekuj da ce
                # ga obarati, pa ne pravi veliki lager
                if self.USE_COMP_STOCK and self.comp_finished.get(good, 0) > (
                    self.COMP_STOCK_RATIO * max(self.Q[good], 1.0)
                ):
                    buffer = min(buffer, self.COMP_STOCK_BUFFER)
                target = ceil(self.Q[good] * buffer)
            # ne proizvodi ako se ne isplati
            if not self.discovery[good] and self.R[good] < cost[good] * self.MIN_MARGIN:
                target = 0
            production[good] = max(0, target - stock)
        return production

    def _make_bids(self, state, lots, recipe_for, cost, production):
        need: dict[str, int] = {}
        for good, qty in production.items():
            for material, amount in recipe_for[good][1].inputs.items():
                need[material] = need.get(material, 0) + amount * qty

        # gornja granica: koliko materijal vredi u najboljem proizvodu
        value_cap: dict[str, float] = {}
        for good, qty in production.items():
            if qty <= 0:
                continue
            recipe = recipe_for[good][1]
            for material, amount in recipe.inputs.items():
                others = cost[good] - self.last_reserve.get(material, 0.0) * amount
                value = (self.R[good] * self.TIER_LOW - others) / amount
                value_cap[material] = max(value_cap.get(material, 0.0), value)

        plan: list[tuple[str, int, float]] = []
        for material, amount in need.items():
            lot = lots.get(material)
            if lot is None or lot.quantity <= 0:
                continue
            qty = min(amount - state.inventory.get(material, 0), lot.quantity)
            if qty <= 0:
                continue
            m = self.m.setdefault(material, self.M_START)
            price = lot.reserve_price * m
            # (b) vidimo po kojoj ceni su drugi pobedili, a nama je falilo
            if (
                self.USE_VISIBLE_BIDS
                and self.fill_short.get(material)
                and material in self.foreign_bid_max
            ):
                price = max(price, self.foreign_bid_max[material] * self.VISIBLE_BID_STEP)
            # (d) konkurenti su prosli put pokupili veliki deo lota
            if self.USE_COMP_RAW and self.comp_raw_gain.get(material, 0) >= (
                self.COMP_RAW_PRESSURE * lot.quantity
            ):
                price *= self.COMP_RAW_BUMP
            cap = value_cap.get(material, 0.0) * self.VALUE_CAP_SHARE
            if cap > lot.reserve_price:
                price = min(price, cap)
                self.m[material] = min(m, cap / lot.reserve_price)
            else:
                price = lot.reserve_price * self.M_MIN
            plan.append((material, qty, price))

        budget = state.capital * self.BUDGET_SHARE
        total = sum(q * p for _, q, p in plan)
        scale = min(1.0, budget / total) if total > 0 else 1.0
        return [
            Bid(material, int(q * scale), round(p, 4))
            for material, q, p in plan
            if int(q * scale) > 0
        ]

    def _make_listings(self, state, goods_config, cost, production):
        listings: list[Listing] = []
        for good in goods_config.finished_goods:
            n = state.inventory.get(good, 0) + production.get(good, 0)
            if n <= 0:
                continue
            R = self.R[good]
            floor = cost.get(good, 0.0) * self.LISTING_FLOOR

            if self.discovery[good]:
                tiers = [(R * t, 1) for t in self.DISCOVERY_TIERS[:n]]
            elif n == 1:
                tiers = [(R * self.TIER_MID, 1)]
            elif n == 2:
                tiers = [(R * self.TIER_MID, 1), (R * self.TIER_HIGH, 1)]
            elif n == 3:
                tiers = [
                    (R * self.TIER_MID, 1),
                    (R * self.TIER_HIGH, 1),
                    (R * self.TIER_LOW, 1),
                ]
            else:
                rest = n - 1
                overstock = n > 2 * self.Q[good] + 2
                low_mult = self.OVERSTOCK_LOW_MULT if overstock else self.TIER_LOW
                low_share = self.OVERSTOCK_LOW_SHARE if overstock else self.SHARE_LOW
                low = round(rest * low_share)
                high = round(rest * self.SHARE_HIGH)
                mid = rest - low - high
                tiers = [
                    (R * self.TIER_MID, mid),
                    (R * self.TIER_PROBE, 1),
                    (R * self.TIER_HIGH, high),
                    (R * low_mult, low),
                ]

            for price, qty in tiers:
                if qty > 0:
                    listings.append(Listing(good, qty, round(max(price, floor), 4)))
        return listings
