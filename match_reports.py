"""PDF, Excel y CSV postpartido desde tracking JSON.

No detecta eventos; exporta los etiquetados, y marca las métricas estimadas.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


def _safe(value):
    return "s/d" if value is None else str(value)


def _player_rows(players):
    return [[p.get("track_id"), p.get("team"), p.get("visible_s"),
             p.get("distance_m"), p.get("avg_speed_kmh"),
             p.get("max_speed_kmh"), p.get("sprints"),
             p.get("accelerations"), p.get("field_coverage_pct")]
            for p in players]


def _tactical_rows(periods):
    rows = []
    for p in periods:
        label = f"{int(p.get('period_start_s', 0) // 60)}-{int(p.get('period_end_s', 0) // 60)}'"
        for team in ("local", "visitante"):
            r = (p.get("teams") or {}).get(team) or {}
            rows.append([label, team, r.get("usable_frames"),
                         r.get("visible_players"), r.get("width_m"),
                         r.get("depth_m"), r.get("area_m2"),
                         r.get("defensive_line_m"),
                         r.get("defenders_near_ball_10m")])
    return rows


def _events_rows(events):
    rows = []
    for event in events:
        if isinstance(event, dict):
            rows.append([event.get("time_s", event.get("t", event.get("start_seconds"))),
                         event.get("event", event.get("type")),
                         event.get("team"), event.get("summary", event.get("description"))])
    return rows


def _write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _pdf_heatmap(samples, codes, title):
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import Color, HexColor

    width, height = 240, 150
    drawing = Drawing(width + 10, height + 35)
    drawing.add(String(10, height + 21, title, fontSize=10))
    cells = [[0] * 4 for _ in range(6)]
    for sample in samples:
        for p in sample.get("pp") or []:
            if len(p) < 3 or p[2] not in codes:
                continue
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if 0 <= x <= 105 and 0 <= y <= 68:
                cells[min(5, int(x / 105 * 6))][min(3, int(y / 68 * 4))] += 1
    peak = max(max(col) for col in cells) or 1
    color = HexColor("#2563EB" if 0 in codes else "#E45757")
    for i in range(6):
        for j in range(4):
            a = cells[i][j] / peak
            c = Color(1 - (1 - color.red) * a,
                      1 - (1 - color.green) * a,
                      1 - (1 - color.blue) * a)
            drawing.add(Rect(10 + i * width / 6, 10 + j * height / 4,
                             width / 6, height / 4, fillColor=c,
                             strokeColor=HexColor("#DDDDDD"), strokeWidth=.4))
    drawing.add(Rect(10, 10, width, height, fillColor=None,
                     strokeColor=HexColor("#777777"), strokeWidth=1))
    return drawing


def _write_pdf(path, data, players, periods, events):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MatchReportTitle", parent=styles["Title"],
                              fontSize=21, leading=25, textColor=colors.HexColor("#14304B")))
    styles.add(ParagraphStyle(name="MatchReportCaveat", parent=styles["BodyText"],
                              fontSize=8, leading=11, textColor=colors.HexColor("#667085")))
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            rightMargin=16 * mm, leftMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    width = A4[0] - 32 * mm
    story = []
    teams = data.get("teams") or {}
    local, away = teams.get("local", "Local"), teams.get("visitante", "Visitante")
    story.append(Paragraph("INFORME POSTPARTIDO", styles["MatchReportTitle"]))
    story.append(Paragraph(escape(f"{local} vs {away}"), styles["Heading2"]))
    story.append(Paragraph(escape(f"Vídeo: {data.get('video_filename', 'Sin especificar')}"),
                           styles["BodyText"]))
    duration = round(float(data.get("match_duration_s") or 0) / 60, 1)
    story.append(Paragraph(
        f"Duración analizada: {duration} min · IDs detectados: {len(players)}",
        styles["BodyText"]))
    story.append(Spacer(1, 12))
    possession = (data.get("stats") or {}).get("posesion") or {}
    local_pct, away_pct = possession.get("local_pct"), possession.get("visitante_pct")
    if local_pct is not None and away_pct is not None:
        story.append(Paragraph("Posesión estimada por equipo", styles["Heading2"]))
        poss = Table([[local, away], [f"{local_pct}%", f"{away_pct}%"]],
                     colWidths=[width / 2, width / 2])
        poss.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EFF6")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#14304B")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        story.append(poss)

    def table(headers, rows, widths):
        if not rows:
            return Paragraph("Sin datos disponibles.", styles["MatchReportCaveat"])
        data_rows = [[Paragraph(escape(_safe(value)), styles["MatchReportCaveat"])
                      for value in line] for line in [headers, *rows]]
        t = Table(data_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EFF6")),
            ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#D5DDE5")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    story.append(Spacer(1, 14))
    story.append(Paragraph("Rendimiento individual (IDs de tracking)", styles["Heading2"]))
    ranked = sorted(players, key=lambda p: p.get("visible_s") or 0, reverse=True)
    rows = [[p.get("track_id"), p.get("team"), p.get("visible_s"),
             p.get("distance_m"), p.get("max_speed_kmh"),
             p.get("field_coverage_pct")] for p in ranked[:25]]
    story.append(table(["ID", "Equipo", "Visible s", "Dist. m", "V. máx", "Campo %"],
                       rows, [width * n for n in (.08, .23, .17, .16, .17, .19)]))
    story.append(Spacer(1, 11))
    story.append(Paragraph(
        "Distancia y velocidades son estimaciones; s/d indica ausencia de datos medibles.",
        styles["MatchReportCaveat"]))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Ocupación territorial (detecciones visibles)", styles["Heading2"]))
    story.append(Table([[
        _pdf_heatmap(data.get("samples") or [], {0, 2}, local),
        _pdf_heatmap(data.get("samples") or [], {1, 3}, away)
    ]], colWidths=[width / 2, width / 2]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Indicadores tácticos por tramos", styles["Heading2"]))
    tactical = _tactical_rows(periods)
    story.append(table(["Tramo", "Equipo", "Jug.", "Ancho", "Prof.", "Área"],
                       [[row[i] for i in (0, 1, 3, 4, 5, 6)] for row in tactical[:36]],
                       [width * n for n in (.16, .24, .13, .14, .14, .19)]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Eventos etiquetados", styles["Heading2"]))
    story.append(table(["Tiempo", "Tipo", "Equipo", "Descripción"],
                       _events_rows(events)[:25],
                       [width * n for n in (.15, .2, .19, .46)]))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        "Advertencia: los identificadores no son dorsales confirmados. "
        "La cámara puede ocultar jugadores. No son mediciones GPS ni eventos verificados.",
        styles["MatchReportCaveat"]))
    doc.build(story)


def _write_xlsx(path, data, player_rows, tactical_rows, event_rows):
    import xlsxwriter
    book = xlsxwriter.Workbook(str(path), {
        "strings_to_formulas": False, "strings_to_urls": False,
    })
    try:
        title = book.add_format({"bold": True, "font_size": 17,
                                 "font_color": "#14304B"})
        head = book.add_format({"bold": True, "bg_color": "#DCE9F5",
                                "border": 1, "text_wrap": True})
        cell = book.add_format({"text_wrap": True, "valign": "top"})
        overview = book.add_worksheet("Resumen")
        overview.set_column("A:A", 28)
        overview.set_column("B:B", 38)
        overview.write("A1", "Match_view - Informe postpartido", title)
        teams = data.get("teams") or {}
        info = [
            ("Equipo local", teams.get("local", "Local")),
            ("Equipo visitante", teams.get("visitante", "Visitante")),
            ("Vídeo", data.get("video_filename", "")),
            ("Duración (min)", round(float(data.get("match_duration_s") or 0) / 60, 1)),
            ("Posesión local (%)", (data.get("stats") or {}).get("posesion", {}).get("local_pct")),
            ("Posesión visitante (%)", (data.get("stats") or {}).get("posesion", {}).get("visitante_pct")),
        ]
        for index, (label, value) in enumerate(info, start=2):
            overview.write(index, 0, label)
            if value is not None:
                overview.write(index, 1, value)
        overview.write("A12", "Estimaciones de tracking, no GPS ni dorsales verificados.")

        def sheet(name, headers, rows, widths):
            worksheet = book.add_worksheet(name)
            worksheet.freeze_panes(1, 0)
            for col, (header, width) in enumerate(zip(headers, widths)):
                worksheet.write(0, col, header, head)
                worksheet.set_column(col, col, width, cell)
            for index, row in enumerate(rows, start=1):
                for col, value in enumerate(row):
                    if value is not None:
                        worksheet.write(index, col, value)
            if rows:
                worksheet.autofilter(0, 0, len(rows), len(headers) - 1)
            return worksheet

        worksheet = sheet("Jugadores", [
            "ID", "Equipo", "Visible (s)", "Distancia (m)",
            "Vel. media (km/h)", "Vel. máxima (km/h)",
            "Sprints", "Aceleraciones", "Campo válido (%)"
        ], player_rows, [8, 18, 15, 18, 20, 22, 12, 15, 19])
        sheet("Táctica", [
            "Tramo", "Equipo", "Muestras", "Jug. visibles", "Ancho m",
            "Prof. m", "Área m2", "Línea def. m", "Defensores cerca balón"
        ], tactical_rows, [15, 18, 14, 16, 15, 15, 17, 18, 25])
        sheet("Eventos", ["Tiempo", "Tipo", "Equipo", "Descripción"],
              event_rows, [15, 20, 20, 55])
        if player_rows:
            chart = book.add_chart({"type": "bar"})
            n = min(len(player_rows), 12)
            chart.add_series({
                "name": "Distancia estimada (m)",
                "categories": ["Jugadores", 1, 0, n, 0],
                "values": ["Jugadores", 1, 3, n, 3],
            })
            chart.set_title({"name": "Distancia por ID visible"})
            chart.set_x_axis({"name": "Metros estimados"})
            chart.set_legend({"none": True})
            overview.insert_chart("D3", chart, {"x_scale": 1.25, "y_scale": 1.20})
    finally:
        book.close()


def generate_match_reports(tracking_json, output_dir=None):
    """Escribe PDF, XLSX y dos CSV a partir de un tracking JSON."""
    tracking_json = Path(tracking_json)
    data = json.loads(tracking_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError("JSON de tracking inválido: falta samples")
    out = Path(output_dir) if output_dir is not None else tracking_json.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = tracking_json.stem
    players = ((data.get("player_stats") or {}).get("players") or [])
    periods = ((data.get("tactical_analysis") or {}).get("periods") or [])
    events = data.get("events") or []
    player_rows, tactic_rows, event_rows = (
        _player_rows(players), _tactical_rows(periods), _events_rows(events))
    paths = {
        "report_pdf": out / f"{stem}_informe.pdf",
        "report_xlsx": out / f"{stem}_estadisticas.xlsx",
        "players_csv": out / f"{stem}_jugadores.csv",
        "tactical_csv": out / f"{stem}_tactica.csv",
    }
    _write_csv(paths["players_csv"], [
        "ID", "Equipo", "Visible s", "Distancia m", "Velocidad media km/h",
        "Velocidad maxima km/h", "Sprints", "Aceleraciones", "Cobertura campo %"
    ], player_rows)
    _write_csv(paths["tactical_csv"], [
        "Tramo", "Equipo", "Muestras", "Jugadores visibles", "Anchura m",
        "Profundidad m", "Area m2", "Linea defensiva m",
        "Defensores a 10m balon"
    ], tactic_rows)
    _write_pdf(paths["report_pdf"], data, players, periods, events)
    _write_xlsx(paths["report_xlsx"], data, player_rows, tactic_rows, event_rows)
    return {key: str(value) for key, value in paths.items()}
