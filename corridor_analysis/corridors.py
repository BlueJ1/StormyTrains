"""The study corridors: three into München Hbf, plus Hamburg Hbf to Frankfurt (Main) Hbf.

Route families group the recorded stop patterns by the stations a run passes.
They are our own grouping for description, not a DB category. `family` returns
the first matching family key; the order of `families` is the display order.
`enroute` names the intermediate stations whose delay is compared with the
final outcome, and `main_family` the family used for the delay build-up chart.
"""

MUNICH = "München Hbf"
# The corridors compared on the comparison page, all ending at München Hbf.
COMPARED = ["berlin", "hamburg", "frankfurt"]


def _berlin(route: set[str]) -> str:
    if "Frankfurt (Main) Hbf" in route:
        return "west" if {"Kassel-Wilhelmshöhe", "Wolfsburg Hbf"} & route else "frankfurt"
    if "Nürnberg Hbf" in route:
        return "augsburg" if "Augsburg Hbf" in route else "direct"
    return "other"


def _hamburg(route: set[str]) -> str:
    if "Berlin Hauptbahnhof" in route:
        return "berlin" if "Nürnberg Hbf" in route else "other"
    if {"Köln Hbf", "Köln Messe/Deutz", "Dortmund Hbf"} & route:
        return "ruhr"
    if "Frankfurt (Main) Hbf" in route:
        return "frankfurt"
    if {"Hannover Hbf", "Würzburg Hbf"} <= route:
        return "hannover"
    return "other"


def _frankfurt(route: set[str]) -> str:
    if "Kassel-Wilhelmshöhe" in route:
        return "other"
    if {"Würzburg Hbf", "Nürnberg Hbf"} <= route:
        return "wuerzburg"
    if {"Stuttgart Hbf", "Ulm Hbf"} & route:
        return "stuttgart"
    return "other"


def _hamburg_frankfurt(route: set[str]) -> str:
    if {"Berlin Hauptbahnhof", "Berlin-Spandau"} & route:
        return "berlin"
    if {"Köln Hbf", "Köln Messe/Deutz", "Dortmund Hbf", "Essen Hbf", "Düsseldorf Hbf",
            "Koblenz Hbf", "Mainz Hbf"} & route:
        return "koeln"
    if "Gießen" in route:
        return "giessen"
    if "Hannover Hbf" in route and not {"Bremen Hbf", "Würzburg Hbf"} & route:
        return "hannover"
    return "other"


CORRIDORS = {
    "berlin": {
        "origin": "Berlin Hauptbahnhof", "label": "Berlin", "short": "Berlin Hbf",
        "destination": MUNICH, "dest_label": "München",
        "family": _berlin, "main_family": "direct",
        "enroute": ["Erfurt Hbf", "Nürnberg Hbf"],
        "families": {
            "direct": ("High-speed via Erfurt and Nürnberg",
                       "Halle or Leipzig, Erfurt, Nürnberg, sometimes Bamberg or Ingolstadt"),
            "augsburg": ("Via Nürnberg and Augsburg",
                         "As above to Nürnberg, then Augsburg and München-Pasing"),
            "frankfurt": ("Via Frankfurt and Stuttgart",
                          "Leipzig, Erfurt, Fulda, Frankfurt, Mannheim, Stuttgart, Ulm, Augsburg"),
            "west": ("Via Kassel and Frankfurt",
                     "Spandau, Wolfsburg, Braunschweig, Göttingen, Kassel, Fulda, Frankfurt, Stuttgart"),
            "other": ("Diversions and rare patterns",
                      "For example via Würzburg skipping Nürnberg, or via the Ruhr and Köln"),
        },
    },
    "hamburg": {
        "origin": "Hamburg Hbf", "label": "Hamburg", "short": "Hamburg Hbf",
        "destination": MUNICH, "dest_label": "München",
        "family": _hamburg, "main_family": "hannover",
        "enroute": ["Kassel-Wilhelmshöhe", "Nürnberg Hbf"],
        "families": {
            "hannover": ("Via Hannover and Würzburg",
                         "Harburg, Hannover, Göttingen, Kassel, Fulda, Würzburg, then Nürnberg or Augsburg"),
            "berlin": ("Via Berlin and Erfurt",
                       "Berlin-Spandau, Berlin Hbf, Südkreuz, Leipzig or Halle, Erfurt, Nürnberg"),
            "ruhr": ("Via Bremen, the Ruhr and Köln",
                     "Bremen, Osnabrück, Münster, Dortmund, Köln, Mannheim, Stuttgart, Ulm, Augsburg"),
            "frankfurt": ("Via Kassel and Frankfurt",
                          "Hannover, Göttingen, Kassel, Frankfurt, Mannheim, Stuttgart, Ulm, Augsburg"),
            "other": ("Diversions and rare patterns",
                      "For example via Berlin and Frankfurt, or skipping Hannover"),
        },
    },
    "frankfurt": {
        "origin": "Frankfurt (Main) Hbf", "label": "Frankfurt", "short": "Frankfurt (Main) Hbf",
        "destination": MUNICH, "dest_label": "München",
        "family": _frankfurt, "main_family": "wuerzburg",
        "enroute": ["Würzburg Hbf", "Ulm Hbf"],
        "families": {
            "wuerzburg": ("Via Würzburg and Nürnberg",
                          "Hanau or Aschaffenburg, Würzburg, Nürnberg, sometimes Ingolstadt"),
            "stuttgart": ("Via Stuttgart and Augsburg",
                          "Mannheim, Mainz or Darmstadt and Heidelberg, Stuttgart, Ulm, Augsburg"),
            "other": ("Diversions and rare patterns",
                      "For example via Würzburg skipping Nürnberg, or a loop through the Ruhr and Kassel"),
        },
    },
    "hamburg_frankfurt": {
        "origin": "Hamburg Hbf", "label": "Hamburg", "short": "Hamburg Hbf",
        "destination": "Frankfurt (Main) Hbf", "dest_label": "Frankfurt",
        "family": _hamburg_frankfurt, "main_family": "hannover",
        "enroute": ["Hannover Hbf", "Kassel-Wilhelmshöhe"],
        "families": {
            "hannover": ("High-speed via Hannover and Kassel",
                         "Harburg or Lüneburg and Uelzen, Hannover, Göttingen, Kassel, sometimes Fulda or Hanau"),
            "giessen": ("Via Kassel and Gießen",
                        "As above to Kassel, then the older line through Marburg and Gießen"),
            "koeln": ("Via Bremen, the Ruhr and Köln",
                      "Bremen, Osnabrück, Münster, the Ruhr, Köln, then the high-speed line or the Rhine valley "
                      "via Koblenz and Mainz"),
            "berlin": ("Via Berlin and Erfurt",
                       "Berlin-Spandau, Berlin Hbf, Südkreuz, Leipzig or Halle, Erfurt, Fulda"),
            "other": ("Diversions and rare patterns",
                      "Mostly via Bremen and Hannover; a few runs recorded with almost no stops"),
        },
    },
}
