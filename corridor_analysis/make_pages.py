"""Build the Hamburg, Frankfurt, Hamburg–Frankfurt and comparison pages from stats/ into pages/.

The Berlin page was published separately before this folder existed and is not
rebuilt here; its numbers are identical to stats/berlin.json.
Figures in the text are filled in from the statistics. The sentences that
interpret them were written for the current data and should be reread after
the data changes.

Run from the project root: .venv/bin/python corridor_analysis/make_pages.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = HERE / "stats"
TEMPLATES = HERE / "templates"
PAGES = HERE / "pages"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def hm(minutes: float) -> str:
    return f"{int(minutes // 60)} h {int(round(minutes % 60)):02d}"


def fmt(n: int) -> str:
    return f"{n:,}"


MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
NOTABLE_DROPPED = ["Frankfurt am Main Flughafen Fernbahnhof", "Erlangen", "Treuchtlingen", "Celle Pbf",
                   "Siegburg/Bonn", "Montabaur", "Coburg", "Günzburg"]


def month_name(month: str) -> str:
    return f"{MONTH_NAMES[int(month[5:]) - 1]} {month[:4]}"


def items(pairs: list[tuple[str, str]]) -> str:
    return "\n    ".join(f"<li><b>{head}</b> {body}</li>" for head, body in pairs)


def questions(pairs: list[tuple[str, str]]) -> str:
    return "\n    ".join(f"<div><h3>{head}</h3><p>{body}</p></div>" for head, body in pairs)


def common_text(R: dict, dropped: list[str] | None = None) -> dict:
    """Text that follows the same pattern on every corridor page. `dropped`
    names the example stations for the panel-filter note."""
    o, c, q = R["overview"], R["cancel"], R["quality"]
    fam = {f["key"]: f for f in R["families"]}
    canceled = c["both"] + c["dest_only"] + c["origin_only"]
    enroute = R["enroute"]
    short = R["short"].replace(" Hauptbahnhof", " Hbf")
    dest = R["dest_label"]
    return {
        "label": R["label"], "origin": R["origin"], "short": short,
        "dest_label": dest, "dest_full": R["destination"],
        "fig_journeys": fmt(o["journeys"]),
        "fig_journeys_text": f"journeys over {o['days']} days, about {round(o['per_day'])} a day, "
                             f"by {o['train_numbers']} train numbers",
        "fig_punctual": pct(o["punctual_6"]),
        "fig_median": f"{o['delay_median']:.0f} min",
        "fig_median_text": f"median arrival delay; the mean is {o['delay_mean']} min because of a long tail "
                           f"up to {o['dmax']:.0f} min",
        "fig_soc": pct(o["severe_or_canceled"]),
        "main_label": fam[R["main_family"]]["label"][0].lower() + fam[R["main_family"]]["label"][1:],
        "main_n": fmt(R["main_n"]),
        "p_corr": "This matters for the <em>during the journey</em> setting. The delay a train already has "
                  "explains a growing part of its final delay: the correlation with the arrival delay at "
                  f"{dest} is <b class=\"mono\">{enroute[0]['corr']:.2f}</b> at {short}, "
                  + ", ".join(f"<b class=\"mono\">{e['corr']:.2f}</b> at {e['station'].replace(' Hbf', '')}"
                              for e in enroute[1:-1])
                  + f" and <b class=\"mono\">{enroute[-1]['corr']:.2f}</b> at "
                  f"{enroute[-1]['station'].replace(' Hbf', '')}. A train leaving "
                  f"{enroute[-1]['station'].replace(' Hbf', '')} an hour or more late ends up severe or canceled "
                  f"{pct(enroute[-1]['rows'][4]['soc'], 0)} of the time.",
        "h_cancel": f"{pct(o['canceled'])} of journeys are canceled",
        "p_cancel": f"Of the {fmt(canceled)} canceled journeys, {fmt(c['both'])} are canceled at both ends. "
                    f"Another {fmt(c['dest_only'])} set off from {short} but never reached {dest}, and "
                    f"{fmt(c['origin_only'])} were canceled at {short} but ran later in the journey. A further "
                    f"{fmt(c['intermediate_only'])} journeys lost only intermediate stops. These are not counted "
                    "as canceled under our definition, but they would be under a stricter one.",
        "_quality": [
            ("Recorded stops changed on 3 November 2025.",
             "Before that date the source only recorded about the 100 largest stations. "
             f"{pct(q['gap_before'])} of earlier journeys have gaps in their stop numbering, against "
             f"{pct(q['gap_after'])} afterwards. The dataset keeps only stations present in every month, so "
             "the set of stations is stable, but stop-level features from before and after that date should "
             "still be compared with care."),
            ("Stations missing from some months are dropped.",
             f"{fmt(R['panel']['stops_removed_outside_panel'])} of "
             f"{fmt(R['panel']['stops_before_panel_filter'])} recorded stops are at stations that are not in "
             "every monthly file, for example "
             + ", ".join((dropped or [s for s in R["panel"]["stations_removed_outside_panel"]
                                      if s in NOTABLE_DROPPED]
                          or R["panel"]["stations_removed_outside_panel"])[:4])
             + ". Journeys still pass through them; they are just not in the stop list."),
            ("Some journeys may have no live updates.",
             f"In {fmt(q['no_update_journeys'])} journeys the reported time equals the scheduled time at every "
             "stop. That could be a perfectly punctual train, or a train with no real-time data. "
             f"{fmt(q['zero_delay_arrivals'])} journeys arrive with exactly 0 minutes of delay."),
            ("Some arrivals are very early.",
             f"{fmt(o['early_5'])} completed journeys arrive 5 or more minutes early, one of them "
             f"{-o['early_min']:.0f} minutes early. These are more likely timetable changes that were never "
             "recorded than real early running."),
            ("The source's own delay column is the wrong one for us.",
             "<span class=\"mono\">delay_in_min</span> gives the departure delay when a train continues "
             f"beyond {dest}, so the arrival delay here is computed from the arrival times."),
        ],
    }


def hamburg_text(R: dict) -> dict:
    o = R["overview"]
    fam = {f["key"]: f for f in R["families"]}
    b = {r["station"]: r for r in R["buildup"]}
    months = sorted(R["monthly"], key=lambda m: -m["soc"])
    jan = next(m for m in R["monthly"] if m["month"] == "2026-01")
    share = lambda k: pct(fam[k]["n"] / o["journeys"], 0)
    return {
        "h_outcomes": "Seven in ten arrive within 20 minutes",
        "h_routes": "Four long ways south",
        "p_routes": f"The {fmt(o['journeys'])} journeys follow {o['routes']} recorded stop patterns, twice as many "
                    f"as from Berlin. We grouped them into route families by the stations they pass. {share('hannover')} "
                    f"take the high-speed line via Hannover, Kassel and Würzburg, with a median schedule of "
                    f"{hm(fam['hannover']['sched'])}. {share('berlin')} run via Berlin and the Erfurt line "
                    f"({hm(fam['berlin']['sched'])}), and {share('ruhr')} take the long way via Bremen, the Ruhr and "
                    f"Köln to Stuttgart ({hm(fam['ruhr']['sched'])}).",
        "callout_routes": f"The runs via Berlin are the same trains as in the Berlin corridor: "
                          f"{fmt(fam['berlin']['n'])} Hamburg journeys also count as Berlin journeys. If both "
                          "corridors are used, the same run must not end up in both the training and the test data.",
        "h_time": "January 2026 stands out",
        "p_time": f"The share of severe and canceled journeys swings between {pct(months[-1]['soc'], 0)} and "
                  f"{pct(months[0]['soc'], 0)} from month to month. The worst months are "
                  + ", ".join(month_name(m["month"]) for m in months[:4])
                  + f". In January 2026, {pct(jan['canceled'])} of journeys were canceled, the highest monthly "
                  "cancellation share on any of the three corridors we compared.",
        "sub_hourly": "Afternoon departures at 13, 15 and 17 h are worst, at 18–20%. Early-morning trains do best.",
        "sub_weekday": "Tuesday highest, Sunday lowest.",
        "h_enroute": "Delay builds up to Kassel, then partly recovers",
        "p_enroute": f"For the high-speed family via Hannover and Würzburg, the share of trains at least 6 minutes "
                     f"late doubles from {pct(b['Hamburg Hbf']['p6'], 0)} at Hamburg Hbf to "
                     f"{pct(b['Kassel-Wilhelmshöhe']['p6'], 0)} at Kassel-Wilhelmshöhe. It then falls a little "
                     "south of Kassel, where the schedule absorbs small delays. The share at least 20 minutes late "
                     f"keeps rising all the way, from {pct(b['Hamburg Hbf']['p20'], 0)} to "
                     f"{pct(b['München Hbf']['p20'], 0)}.",
        "h_trains": "The fast line's afternoon trains do worst",
        "p_trains": f"Among the {R['trains_count']} train numbers with at least 300 journeys, the share of severe "
                    f"or canceled runs ranges from {pct(R['trains_best'][0]['soc'], 0)} to "
                    f"{pct(R['trains_worst'][0]['soc'], 0)}. The least reliable are the 13:01, 15:01 and 17:01 "
                    "departures on the line via Hannover and Würzburg. The same line's early-morning trains are "
                    "among the most reliable, so the departure slot seems to matter at least as much as the route.",
        "questions": questions([
            ("Do the runs via Berlin belong to this pair?",
             f"{share('berlin')} of Hamburg journeys go through Berlin and are also Berlin–München journeys. "
             "Keeping them makes the two corridors partly the same data; dropping them removes almost a third of this one."),
            ("Which route families belong in the sample?",
             f"{o['routes']} stop patterns and schedules from {hm(o['sched_p10'])} to {hm(o['sched_p90'])} "
             "(fastest and slowest 10%). The route family is likely to be needed as a feature, or the sample "
             "restricted to the line via Hannover."),
            ("What happened in January 2026?",
             f"{pct(jan['canceled'])} of journeys were canceled that month. Knowing the cause matters before we "
             "decide whether such months belong in training or test data."),
            ("How strong is the during-journey baseline?",
             f"The delay at Nürnberg already correlates {R['enroute'][-1]['corr']:.2f} with the final delay. "
             "The persistence baseline will be hard to beat late in the trip; the useful comparison is early on."),
        ]),
    }


def frankfurt_text(R: dict) -> dict:
    o = R["overview"]
    fam = {f["key"]: f for f in R["families"]}
    b = {r["station"]: r for r in R["buildup"]}
    s = R["origin_start"]
    months = sorted(R["monthly"], key=lambda m: -m["soc"])
    jun = next(m for m in R["monthly"] if m["month"] == "2026-06")
    share = lambda k: pct(fam[k]["n"] / o["journeys"], 0)
    return {
        "h_outcomes": "Only four in ten are punctual",
        "h_routes": "Two main routes of similar length",
        "p_routes": f"The {fmt(o['journeys'])} journeys follow only {o['routes']} recorded stop patterns. "
                    f"{share('wuerzburg')} run via Würzburg and Nürnberg with a median schedule of "
                    f"{hm(fam['wuerzburg']['sched'])}, and {share('stuttgart')} via Mannheim or Darmstadt, "
                    f"Stuttgart and Augsburg ({hm(fam['stuttgart']['sched'])}). The remaining "
                    f"{share('other')} are diversions and a few trains that loop through the Ruhr and Kassel.",
        "callout_routes": f"The two main routes differ by about half an hour and have almost the same share of "
                          f"severe or canceled journeys ({pct(fam['wuerzburg']['soc'])} and "
                          f"{pct(fam['stuttgart']['soc'])}). The route family matters less here than on the "
                          "Berlin or Hamburg corridors.",
        "h_time": "June 2026 was the worst month",
        "p_time": f"The share of severe and canceled journeys swings between {pct(months[-1]['soc'], 0)} and "
                  f"{pct(months[0]['soc'], 0)} from month to month. The worst months are June 2026, July 2025, "
                  "October 2025 and January 2026; the best are the two Decembers. In June 2026, "
                  f"{pct(jun['canceled'])} of journeys were canceled.",
        "sub_hourly": "Late-morning departures at 10 and 11 h are worst, at about 20%. Trains leaving at 01 and "
                      "02 h are night services with schedules of about 5 hours.",
        "_quality_extra": [
            ("Frankfurt Airport is not in the stable station list.",
             "Its long-distance station is missing from some monthly files, so it is dropped from the stop "
             "lists. Trains that stop at the airport but not at Frankfurt Hbf are not in this corridor at all, "
             "and neither are the 219 journeys that pass only Frankfurt (Main) Süd."),
        ],
        "sub_weekday": "Weekdays 13–16%, weekends about 11%.",
        "h_enroute": "Half the trains are already late at Frankfurt",
        "p_enroute": f"{pct(1 - s['starts_here'], 0)} of these trains start before Frankfurt Hbf. "
                     f"{pct(s['late_if_through'], 0)} of them leave Frankfurt at "
                     f"least 6 minutes late, against {pct(s['late_if_starts_here'], 0)} of trains that start there. "
                     f"On the route via Würzburg, the share at least 6 minutes late rises from "
                     f"{pct(b['Frankfurt (Main) Hbf']['p6'], 0)} at Frankfurt to "
                     f"{pct(b['Aschaffenburg Hbf']['p6'], 0)} at Aschaffenburg, only half an hour later, and "
                     f"ends at {pct(b['München Hbf']['p6'], 0)} in München.",
        "h_trains": "Late-morning trains do worst",
        "p_trains": f"Among the {R['trains_count']} train numbers with at least 300 journeys, the share of severe "
                    f"or canceled runs ranges from {pct(R['trains_best'][0]['soc'], 0)} to "
                    f"{pct(R['trains_worst'][0]['soc'], 0)}. The least reliable are the 10:22 and 11:54 departures "
                    "via Würzburg and several afternoon trains via Stuttgart. Early-morning departures via Würzburg "
                    "are among the most reliable.",
        "questions": questions([
            ("What does the model know at departure?",
             f"{pct(1 - s['starts_here'], 0)} of trains are already running when they reach Frankfurt, and half "
             "arrive late. Whether the delay at earlier stops is known at the prediction time is part of the "
             "kickoff decision on the prediction cutoff."),
            ("Is the short corridor a useful contrast?",
             f"At {hm(o['sched_median'])} it is the shortest of the three, with only {o['routes']} stop patterns "
             f"and far fewer gaps in the stop data before November 2025 ({pct(R['quality']['gap_before'], 0)})."),
            ("Where does the test period fall?",
             "June 2026 was the worst month and lies near the end of the data. A test period in summer 2026 "
             "would be harder than the training period."),
            ("How do we handle an imbalanced target?",
             f"Only {pct(o['severe_or_canceled'])} of journeys are severe or canceled. Precision–recall based "
             "metrics and class-balanced baselines fit better than accuracy."),
        ]),
    }


def hamburg_frankfurt_text(R: dict) -> dict:
    o = R["overview"]
    fam = {f["key"]: f for f in R["families"]}
    b = {r["station"]: r for r in R["buildup"]}
    s = R["origin_start"]
    months = sorted(R["monthly"], key=lambda m: -m["soc"])
    jan = next(m for m in R["monthly"] if m["month"] == "2026-01")
    share = lambda k: pct(fam[k]["n"] / o["journeys"], 0)
    shared = R["shared_runs"]["hamburg"]
    return {
        "h_outcomes": "Seven in ten within 20 minutes, four in ten punctual",
        "h_routes": "One fast line and three slower ones",
        "p_routes": f"The {fmt(o['journeys'])} journeys follow {o['routes']} recorded stop patterns. "
                    f"{share('hannover')} take the high-speed line via Hannover and Kassel, with a median schedule "
                    f"of {hm(fam['hannover']['sched'])}. {share('giessen')} leave that line after Kassel for the "
                    f"older route through Gießen ({hm(fam['giessen']['sched'])}), {share('koeln')} go west via "
                    f"Bremen, the Ruhr and Köln ({hm(fam['koeln']['sched'])}), and {share('berlin')} via Berlin "
                    f"and Erfurt ({hm(fam['berlin']['sched'])}).",
        "callout_routes": f"The routes fail in different ways. Via Köln the median arrival delay is "
                          f"{fam['koeln']['median']:.0f} min, against {fam['hannover']['median']:.0f} min on the "
                          f"high-speed line, and {pct(fam['koeln']['moderate'], 0)} of journeys are moderately late. "
                          f"Via Gießen the median delay is only {fam['giessen']['median']:.0f} min, but "
                          f"{pct(fam['giessen']['canc'], 0)} of journeys are canceled, and the route has almost no "
                          "runs after December 2025.",
        "h_time": "January 2026 was the worst month",
        "p_time": f"The share of severe and canceled journeys swings between {pct(months[-1]['soc'], 0)} and "
                  f"{pct(months[0]['soc'], 0)} from month to month. The worst months are "
                  + ", ".join(month_name(m["month"]) for m in months[:4])
                  + "; the best are "
                  + ", ".join(month_name(m["month"]) for m in months[:-4:-1])
                  + f". In January 2026, {pct(jan['canceled'])} of journeys were canceled.",
        "sub_hourly": "Departures at 11 and 16 h are worst, at about 20%. Early-morning trains do best.",
        "sub_weekday": "Weekdays about 15%, weekends under 12%.",
        "h_enroute": "Most of the delay builds up before Hannover",
        "p_enroute": f"On the high-speed line, the share of trains at least 6 minutes late rises from "
                     f"{pct(b['Hamburg Hbf']['p6'], 0)} at Hamburg Hbf to {pct(b['Hannover Hbf']['p6'], 0)} at "
                     f"Hannover, in the first hour and a quarter, and then more slowly to "
                     f"{pct(b['Frankfurt (Main) Hbf']['p6'], 0)} at Frankfurt. The share at least 20 minutes late "
                     f"grows from {pct(b['Hamburg Hbf']['p20'], 0)} to {pct(b['Frankfurt (Main) Hbf']['p20'], 0)}. "
                     f"{pct(1 - s['starts_here'], 0)} of all runs start before Hamburg Hbf, but unlike on the "
                     f"Frankfurt–München corridor this hardly matters: {pct(s['late_if_through'], 0)} of them leave Hamburg at least 6 minutes "
                     f"late, against {pct(s['late_if_starts_here'], 0)} of trains that start there.",
        "h_trains": "Gießen and Köln trains do worst",
        "p_trains": f"Among the {R['trains_count']} train numbers with at least 300 journeys, the share of severe "
                    f"or canceled runs ranges from {pct(R['trains_best'][0]['soc'], 0)} to "
                    f"{pct(R['trains_worst'][0]['soc'], 0)}. Three of the six least reliable run via Gießen: they "
                    "are rarely late when they run, but often canceled. Two run via Köln, with median delays over "
                    "30 minutes. All six most reliable trains use the high-speed line and leave Hamburg between "
                    f"{min(t['dep'] for t in R['trains_best'])} and {max(t['dep'] for t in R['trains_best'])}.",
        "_quality_extra": [
            ("Only Frankfurt (Main) Hbf counts as the destination.",
             "5,339 further Hamburg Hbf runs reach Frankfurt only at the airport or at Frankfurt (Main) Süd "
             "and are not in this corridor. The airport station is also missing from some monthly files, so it "
             "is dropped from the stop lists."),
        ],
        "questions": questions([
            ("Which runs are shared with the München corridors?",
             f"{fmt(shared)} runs ({pct(shared / o['journeys'], 0)}) continue from Frankfurt to München and are "
             "also Hamburg–München and Frankfurt–München journeys. A split into training and test data has to keep "
             "each run on one side."),
            ("What happens to the Gießen route?",
             f"It carries {share('giessen')} of all journeys, almost all of them before January 2026. A test "
             "period in 2026 would contain a different mix of routes than the training period."),
            ("Should the destination include the airport and Süd?",
             "Counting them would add about a quarter more runs, most of them at the airport. Whether they belong "
             "depends on whether the question is about Frankfurt Hbf or about reaching Frankfurt at all."),
            ("How strong is the during-journey baseline?",
             f"The delay at Kassel, about 80 minutes before Frankfurt, already correlates "
             f"{R['enroute'][-1]['corr']:.2f} with the final delay. A model has most room to add value at "
             "departure and around Hannover."),
        ]),
    }


def comparison_text(C: dict) -> dict:
    K = {c["key"]: c for c in C["corridors"]}
    o = {k: c["overview"] for k, c in K.items()}
    ov, mc = C["overlap"], C["monthly_corr"]
    late = {k: sum(r["n"] for r in c["enroute"][0]["rows"][1:]) / c["enroute"][0]["n"] for k, c in K.items()}
    return {
        "lede": "Berlin Hbf, Hamburg Hbf and Frankfurt (Main) Hbf to München Hbf: every ICE and IC run that "
                "stopped at the origin and later at München, 1 July 2024 to 31 August 2026. The same rules and "
                "severity classes on all three, railway data only, no weather.",
        "h_outcomes": "Similar shares of bad journeys, different kinds of lateness",
        "p_outcomes": f"The severe-or-canceled share is close on all three corridors: {pct(o['berlin']['severe_or_canceled'])} "
                      f"from Berlin, {pct(o['hamburg']['severe_or_canceled'])} from Hamburg and "
                      f"{pct(o['frankfurt']['severe_or_canceled'])} from Frankfurt. Frankfurt, the shortest trip, has "
                      f"the fewest punctual arrivals ({pct(o['frankfurt']['punctual_6'])}) and a median delay of "
                      f"{o['frankfurt']['delay_median']:.0f} min: more trains are moderately late, but no more are "
                      "severely late than elsewhere. "
                      f"Hamburg has the most cancellations ({pct(o['hamburg']['canceled'])}).",
        "h_routes": "Frankfurt is short and simple; Hamburg is long and varied",
        "p_routes": f"From Frankfurt, {o['frankfurt']['routes']} stop patterns fall into two main routes of similar "
                    f"length. From Hamburg there are {o['hamburg']['routes']} patterns, and the slowest 10% of runs "
                    f"take {hm(o['hamburg']['sched_p90'])} or more. Berlin sits in between, with one dominant "
                    "high-speed route and a long tail of detours.",
        "h_time": "Bad months are largely shared",
        "p_time": f"The monthly severe-or-canceled shares move together: the correlation is {mc['berlin-hamburg']:.2f} "
                  f"between Berlin and Hamburg, {mc['hamburg-frankfurt']:.2f} between Hamburg and Frankfurt, and "
                  f"{mc['berlin-frankfurt']:.2f} between Berlin and Frankfurt. July 2025 and January 2026 are bad on "
                  "all three. Part of this is shared trains, part is whatever affects the whole network at once.",
        "h_enroute": "Frankfurt trains bring their delay with them",
        "p_enroute": f"{pct(late['frankfurt'], 0)} of trains leave Frankfurt at least 6 minutes late, against "
                     f"{pct(late['berlin'], 0)} at Berlin and {pct(late['hamburg'], 0)} at Hamburg, because most "
                     "Frankfurt runs started further back. So the delay at departure already says a lot about the "
                     f"outcome there (correlation {K['frankfurt']['enroute'][0]['corr']:.2f}), and little at Hamburg "
                     f"({K['hamburg']['enroute'][0]['corr']:.2f}). On all three corridors the correlation climbs to "
                     "about 0.9 once three quarters of the schedule is done.",
        "h_cancel": "Hamburg loses the most journeys",
        "p_cancel": "Most canceled journeys are canceled at both ends. Hamburg has a larger share canceled only at "
                    "the origin, where the train then ran later in the journey, and the most journeys that lost "
                    "only intermediate stops.",
        "h_overlap": "The same run can count on more than one corridor",
        "p_overlap": f"A train from Hamburg via Berlin to München is a Hamburg journey and a Berlin journey. "
                     f"{fmt(ov['berlin-hamburg'])} runs are in both of those corridors, a third of Hamburg's. "
                     f"{fmt(ov['berlin-frankfurt'])} Berlin runs go via Frankfurt and are Frankfurt journeys too. "
                     "One ICE 699 run is even in all three. Their outcomes at München are identical, so any split "
                     "into training and test data has to keep a run on one side.",
        "h_quality": "Frankfurt's stop data is the most complete",
        "p_quality": "Before the source widened its station coverage on 3 November 2025, most Berlin and Hamburg "
                     "journeys have gaps in their stop numbering; far fewer Frankfurt journeys do. After the "
                     "change, gaps are rare everywhere.",
        "questions": questions([
            ("One corridor or two?",
             "Berlin and Hamburg share a third of Hamburg's runs and most of their bad months, so as a pair they "
             "add less independent data. Frankfurt shares fewer runs and behaves differently at departure."),
            ("Does trip length matter for the question?",
             f"Frankfurt ({hm(o['frankfurt']['sched_median'])}) gives a short, simple corridor; Hamburg "
             f"({hm(o['hamburg']['sched_median'])}) a long one with many routes. Weather along a 6-hour trip is "
             "harder to summarise than along a 3-hour one."),
            ("What is known at prediction time?",
             "For Frankfurt, the delay the train brings in is the strongest single signal. Whether it counts as "
             "known before departure depends on the prediction cutoff we choose."),
            ("How much data is enough?",
             f"All three have {fmt(min(x['journeys'] for x in o.values()))} or more journeys and about "
             f"{min(round(x['severe_or_canceled'] * x['journeys'], -2) for x in o.values()):,.0f} or more "
             "positives, so any of them supports the core models."),
        ]),
    }


def build(template: str, data: dict, text: dict) -> str:
    html = (TEMPLATES / template).read_text()
    html = html.replace("{{STYLE}}", (TEMPLATES / "style.css").read_text())
    html = html.replace("{{COMMON}}", (TEMPLATES / "common.js").read_text())
    html = html.replace("{{DATA}}", json.dumps(data, ensure_ascii=False))
    for key, value in text.items():
        html = html.replace("{{" + key + "}}", str(value))
    assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
    return html


def main() -> None:
    PAGES.mkdir(exist_ok=True)
    for key, text, dropped in (("hamburg", hamburg_text, None), ("frankfurt", frankfurt_text, None),
                               ("hamburg_frankfurt", hamburg_frankfurt_text,
                                ["Celle Pbf", "Marburg (Lahn)", "Siegburg/Bonn",
                                 "Frankfurt am Main Flughafen Fernbahnhof"])):
        R = json.loads((STATS / f"{key}.json").read_text())
        page = {**common_text(R, dropped), **text(R)}
        page["quality_items"] = items(page.pop("_quality") + page.pop("_quality_extra", []))
        (PAGES / f"{key}.html").write_text(build("corridor.html", R, page))
    C = json.loads((STATS / "comparison.json").read_text())
    (PAGES / "comparison.html").write_text(build("comparison.html", C, comparison_text(C)))
    print("wrote", sorted(p.name for p in PAGES.glob("*.html")))


if __name__ == "__main__":
    main()
