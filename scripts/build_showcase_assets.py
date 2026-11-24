"""Build branding, the author's workflow figure, and data-backed showcase figures.

Only docs/assets (or an explicit --output-dir) is written. No model, optimizer,
scientific result, or original figure is changed. PNGs are rendered directly
from the included numeric exports rather than resized screenshots or cards.

The default reuses and verifies the committed figure-only workflow PNG and its
source metadata. The complete manuscript is neither needed nor distributed.
To rebuild that PNG from the author's local original PDF, optionally run:
    python scripts/build_showcase_assets.py --workflow-only --workflow-pdf PATH
The fixed crop is Fig. 2 on PDF page 5 and excludes its caption and article text.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
INK = "#143342"
BLUE = "#245B78"
TEAL = "#159F94"
GOLD = "#C67A36"
PALE = "#D8E4E8"
WORKFLOW_PAGE = 5
WORKFLOW_CROP_TOP_LEFT = (70.0, 50.0, 540.0, 414.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workflow_asset(output, source_pdf=None, dpi=400):
    """Use the authorized figure crop, never a complete manuscript page.

    The crop is given in 1/72-inch PDF points, from the top-left of the upright
    page: left, top, right, bottom. Rendering is a direct PDFium page-region
    operation; no raster image is retouched or reinterpreted.
    """
    output.mkdir(parents=True, exist_ok=True)
    target = output / "workflow.png"
    if source_pdf is not None:
        try:
            import pypdfium2 as pdfium
        except ImportError as error:
            raise RuntimeError("Optional PDF extraction requires pypdfium2; the default build does not.") from error
        source_pdf = Path(source_pdf).resolve()
        document = pdfium.PdfDocument(str(source_pdf))
        if len(document) < WORKFLOW_PAGE:
            raise ValueError("The supplied source has no workflow figure on page 5.")
        page = document[WORKFLOW_PAGE - 1]
        width, height = page.get_size()
        left, top, right, bottom = WORKFLOW_CROP_TOP_LEFT
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("The fixed workflow crop does not fit the supplied page.")
        margins = (left, height - bottom, width - right, top)
        bitmap = page.render(scale=dpi / 72.0, crop=margins)
        pixel_size = [bitmap.width, bitmap.height]
        bitmap.to_pil().save(str(target))
        bitmap.close()
        page.close()
        document.close()
        return {
            "file": "workflow.png",
            "mode": "user_authorized_original_figure_crop",
            "description": "The author's complete three-layer workflow schematic, original Fig. 2 panels (a)-(c), cropped without the caption, article text, title or authors.",
            "source_files": {},
            "external_source": {
                "filename": source_pdf.name, "sha256": sha256(source_pdf),
                "figure": "Fig. 2", "pdf_page_one_based": WORKFLOW_PAGE,
                "pdf_page_size_points": [width, height],
                "crop_top_left_points": list(WORKFLOW_CROP_TOP_LEFT),
                "crop_bottom_left_points": [left, height - bottom, right, height - top],
                "coordinate_units": "PDF points (1/72 inch)",
                "render_dpi": dpi,
                "pixel_size": pixel_size,
                "complete_source_pdf_distributed": False,
                "permission": "User explicitly authorized reuse of their own manuscript figure.",
                "interpretation": "This is the original workflow schematic; small numerical examples inside it are schematic labels, not replacements for current run metrics.",
            },
            "rebuild": "python scripts/build_showcase_assets.py --workflow-only --workflow-pdf PATH_TO_LOCAL_ORIGINAL_PDF",
            "sha256": sha256(target),
        }
    # Rebuilding the public repository needs only the committed crop and record.
    # Custom output directories may reuse that same verified crop byte-for-byte.
    for folder in [output, ROOT / "docs/assets"]:
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = next((entry for entry in manifest.get("assets", []) if entry["file"] == "workflow.png"), None)
        if record is not None:
            original = folder / "workflow.png"
            if not original.is_file() or sha256(original) != record["sha256"]:
                raise ValueError("The committed workflow crop does not match its manifest.")
            if original.resolve() != target.resolve():
                shutil.copyfile(original, target)
            return record
    raise FileNotFoundError("Provide the committed workflow.png and manifest, or use --workflow-pdf to rebuild the figure-only crop.")


def update_workflow_only(output, source_pdf=None, dpi=400):
    """Replace only the workflow asset and its metadata; leave galleries intact."""
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = workflow_asset(output, source_pdf, dpi)
    manifest["assets"] = [entry for entry in manifest["assets"] if entry["file"] not in {"workflow.svg", "workflow.png"}]
    manifest["assets"].insert(1, record)
    manifest["preferred_workflow_asset"] = "workflow.png"
    manifest["generator_sha256"] = sha256(Path(__file__))
    manifest["policy"] = "Original branding, an authorized figure-only workflow crop, and plotting of included numerical exports. No complete manuscript is distributed; no scientific source or result is modified."
    manifest["png_canvas"]["applies_to"] = [entry["file"] for entry in manifest["assets"] if entry["file"].endswith(".png") and entry["file"] != "workflow.png"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    obsolete = output / "workflow.svg"
    if obsolete.is_file():
        obsolete.unlink()
    return manifest


def text(x, y, value, size=19, fill=INK, weight="400", family="Arial, sans-serif", extra=""):
    return f'<text x="{x}" y="{y}" fill="{fill}" font-family="{family}" font-size="{size}" font-weight="{weight}" {extra}>{html.escape(value)}</text>'


def hero_svg():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="440" viewBox="0 0 1600 440" role="img" aria-labelledby="title desc">',
         '<title id="title">AI Buckingham</title>',
         '<desc id="desc">Original conceptual artwork: physical quantities become interpretable dimensionless scaling. Decorative curves are illustrative, not experimental data.</desc>',
         '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#082B3B"/><stop offset="1" stop-color="#104C54"/></linearGradient>',
         '<radialGradient id="glow"><stop stop-color="#2BB4A6" stop-opacity=".23"/><stop offset="1" stop-color="#2BB4A6" stop-opacity="0"/></radialGradient></defs>',
         '<rect width="1600" height="440" fill="url(#bg)"/>',
         '<circle cx="1350" cy="190" r="390" fill="url(#glow)"/>',
         '<path d="M0 406H1600" stroke="#71C4BD" stroke-opacity=".26"/>',
         '<path d="M80 64H155" stroke="#4AD4BE" stroke-width="4"/>',
         text(80, 109, "PHYSICS  /  INFORMATION  /  AGENTS", 20, "#8FD5CD", "600", extra='letter-spacing="3"'),
         text(75, 221, "AI Buckingham", 91, "#F7FAFB", "400", "Georgia, Times New Roman, serif"),
         text(80, 280, "Agentic discovery of dimensionless scaling laws", 29, "#D4E7EA"),
         text(80, 320, "across physical regimes.", 29, "#D4E7EA"),
         text(80, 380, "Grounded reasoning", 22, "#F4FAFA", "500"),
         text(326, 380, "·", 24, "#4AD4BE"),
         text(354, 380, "Numerical discovery", 22, "#F4FAFA", "500"),
         text(618, 380, "·", 24, "#4AD4BE"),
         text(646, 380, "Traceable evidence", 22, "#F4FAFA", "500")]
    # This is deliberately conceptual vector artwork, with no numeric labels.
    s += ['<g stroke="#B0DEDC" stroke-opacity=".13" stroke-width="1">']
    for x in range(1020, 1511, 70):
        s.append(f'<path d="M{x} 76V353"/>')
    for y in range(91, 354, 52):
        s.append(f'<path d="M1008 {y}H1520"/>')
    s += ['</g>', '<path d="M1015 350H1525M1015 350V65" fill="none" stroke="#84BBB8" stroke-width="2"/>',
          '<path d="M1045 320C1120 319 1150 290 1197 236S1328 150 1496 110" fill="none" stroke="#50D0BD" stroke-width="4"/>',
          '<path d="M1045 307C1100 318 1158 277 1203 235S1320 156 1496 116" fill="none" stroke="#F1C389" stroke-width="2.5" stroke-dasharray="6 8" opacity=".9"/>',
          '<path d="M1045 326C1110 315 1162 298 1200 242S1330 148 1496 108" fill="none" stroke="#C1E5EE" stroke-width="2.5" stroke-dasharray="2 10" opacity=".85"/>']
    for x, y in [(1067, 318), (1132, 291), (1194, 240), (1264, 190), (1341, 151), (1421, 126)]:
        s.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#0D3D48" stroke="#68DCC9" stroke-width="2"/>')
    s += [text(1460, 385, "Π*", 33, "#D5ECEC", "400", "Georgia, serif"),
          text(1050, 56, "DIMENSIONLESS REPRESENTATION", 16, "#91C3C5", "500", extra='letter-spacing="2"'),
          text(1184, 412, "Π = ∏ xᵢʷⁱ", 27, "#8FD5CD", "400", "Georgia, serif"), '</svg>']
    return "\n".join(s) + "\n"


def selected_run(case_id):
    index = json.loads((ROOT / "results/index.json").read_text(encoding="utf-8"))
    case = next(record for record in index["cases"] if record["case_id"] == case_id)
    run = ROOT / "results/cases" / case_id / case["selected_run"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Showcase requires a completed selected run: {case_id}")
    return run


def build(output, source_pdf=None, workflow_dpi=400):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    records = []

    def save(fig, filename, sources, description, mode="visualization_of_recorded_numeric_exports"):
        fig.savefig(output / filename, dpi=100, facecolor="white")
        plt.close(fig)
        records.append({"file": filename, "mode": mode, "description": description,
                        "source_files": {str(p.relative_to(ROOT)).replace("\\", "/"):sha256(p) for p in sources},
                        "sha256":sha256(output/filename)})

    workflow = workflow_asset(output, source_pdf, workflow_dpi)
    records.append(workflow)
    obsolete = output / "workflow.svg"
    if obsolete.is_file():
        obsolete.unlink()
    for name, markup, description in [("hero.svg",hero_svg(),"Original project branding with illustrative, non-numeric scaling curves.")]:
        (output/name).write_text(markup,encoding="utf-8")
        records.append({"file":name,"mode":"original_repository_vector_artwork","description":description,"source_files":{},"sha256":sha256(output/name)})

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":15,"axes.titlesize":17,
                         "axes.labelsize":15,"axes.labelcolor":INK,"text.color":INK,
                         "xtick.labelsize":13,"ytick.labelsize":13,"axes.edgecolor":"#5E7883",
                         "axes.linewidth":.9,"grid.color":"#D9E4E8","grid.alpha":.65,
                         "legend.fontsize":12,"mathtext.fontset":"dejavusans"})

    def base_figure(title, subtitle):
        fig=plt.figure(figsize=(12,5.5),dpi=100)
        fig.text(.055,.965,title,fontsize=22,fontfamily="DejaVu Serif",color=INK,va="top")
        fig.text(.055,.902,subtitle,fontsize=13,color="#5A737E",va="top")
        return fig

    # MHD: actual exported grid, actual best-so-far optimizer coordinates.
    folder=ROOT/"results/mi_landscapes/mhd"
    grid=pd.read_csv(folder/"response_surface.csv").pivot(index="coefficient_y",columns="coefficient_x",values="mutual_information")
    history=pd.read_csv(folder/"generation_history.csv")
    fig=base_figure("Discovering a dominant dimensionless direction", "MHD · saved MI grid and actual primary-seed optimizer trajectory")
    ax=fig.add_axes([.10,.20,.35,.59])
    contour=ax.contourf(grid.columns.to_numpy(),grid.index.to_numpy(),grid.to_numpy(),levels=24,cmap="viridis")
    ax.plot(history.best_c_0,history.best_c_1,color="white",linewidth=1.8,zorder=4)
    ax.scatter(history.best_c_0.iloc[0],history.best_c_1.iloc[0],s=50,facecolor="white",edgecolor=INK,zorder=5)
    ax.scatter(history.best_c_0.iloc[-1],history.best_c_1.iloc[-1],s=180,marker="*",color="#F6D266",edgecolor=INK,zorder=6)
    ax.set(xlim=(grid.columns.min(),grid.columns.max()),ylim=(grid.index.min(),grid.index.max()),xlabel=r"$c_1$",ylabel=r"$c_2$",title="MI landscape & recorded path")
    cax=fig.add_axes([.465,.20,.017,.59]);fig.colorbar(contour,cax=cax).ax.tick_params(labelsize=11)
    right=fig.add_axes([.575,.16,.38,.66],projection="3d")
    xx,yy=np.meshgrid(grid.columns.to_numpy(),grid.index.to_numpy())
    right.plot_surface(xx,yy,grid.to_numpy(),cmap="viridis",linewidth=0,antialiased=True,rcount=55,ccount=55)
    right.view_init(elev=25,azim=-132)
    right.set(xlabel=r"$c_1$",ylabel=r"$c_2$",zlabel="Estimated MI",title="Same numeric grid")
    right.tick_params(labelsize=10,pad=1);right.xaxis.labelpad=0;right.yaxis.labelpad=0;right.zaxis.labelpad=4
    fig.text(.07,.035,"Finite-sample MI values are retained, including negative estimates.",fontsize=12,color="#5A737E")
    save(fig,"mhd_mi_landscape.png",[folder/"response_surface.csv",folder/"generation_history.csv",folder/"run_metadata.json"],"Two views of the saved MHD grid; the left path uses the actual best-so-far coefficients, not an invented trajectory.")

    # Regime figures: every scatter point comes from the selected run's tables.
    for case_id in ["cess_channel","rough_pipe"]:
        run=selected_run(case_id)
        gradient_path=run/"tables/gradient_field.csv"
        gradient=pd.read_csv(gradient_path)
        expected_counts={"cess_channel":{0:107,1:114,2:779},"rough_pipe":{0:975,1:225}}
        if gradient.regime.value_counts().to_dict()!=expected_counts[case_id]:
            raise ValueError("The selected publication regime labels changed; review the descriptive legend before rebuilding.")
        state_path=run/"state.json"
        state=json.loads(state_path.read_text(encoding="utf-8"))
        local=state["workflow_result"]["local_workflows"]
        files=[gradient_path,state_path,run/"manifest.json"]
        if case_id=="cess_channel":
            title="Gradient regimes, local scaling"
            subtitle="Cess eddy-viscosity model · 1,000 recorded observations · three selected regimes"
            names={0:"Outer",1:"Intermediate",2:"Near-wall"};colors={0:GOLD,1:TEAL,2:BLUE}
            fig=base_figure(title,subtitle)
            left=fig.add_axes([.075,.30,.375,.48]);right=fig.add_axes([.60,.30,.365,.48])
            for regime,label in names.items():
                part=gradient[gradient.regime==regime]
                left.scatter(part.z_1,part.z_2,s=10,color=colors[regime],alpha=.66,rasterized=True)
                arrows=part.iloc[::max(1,len(part)//15)]
                g=arrows[["gradient_z_1","gradient_z_2"]].to_numpy()
                norm=np.linalg.norm(g,axis=1)
                unit=np.divide(g,norm[:,None],out=np.zeros_like(g),where=norm[:,None]>0)
                scale=np.maximum(np.ptp(gradient[["z_1","z_2"]].to_numpy(),axis=0),1e-12)*.045
                left.quiver(arrows.z_1,arrows.z_2,unit[:,0]*scale[0],unit[:,1]*scale[1],angles="xy",scale_units="xy",scale=1,color=INK,width=.0032,headwidth=3.4)
            left.set(xlabel=r"$z_1$ · orthonormal log-$\Pi$",ylabel=r"$z_2$",title="Recorded gradient-based partition")
            response_label=r"$\nu_t/\nu$"
        else:
            title="Regime-wise scaling in rough-pipe flow"
            subtitle="Selected completed run · 1,200 observations · two distinct physical dependences"
            names={0:"Roughness-dominated",1:"Laminar"};colors={0:TEAL,1:BLUE}
            fig=base_figure(title,subtitle)
            left=fig.add_axes([.075,.30,.375,.48]);right=fig.add_axes([.60,.30,.365,.48])
            dataset=ROOT/"data/datasets/rough_pipe.csv"
            data=pd.read_csv(dataset).iloc[gradient.csv_row.to_numpy(dtype=int)].reset_index(drop=True)
            data["regime"]=gradient.regime.to_numpy()
            data["Re"]=data.rho*data.U*data.D/data.mu
            for regime in names:
                part=data[data.regime==regime]
                left.scatter(part.Re,part.f_D,s=11,color=colors[regime],alpha=.65,rasterized=True)
            left.set(xscale="log",yscale="log",xlabel=r"$Re=\rho UD/\mu$",ylabel=r"$f_D$",title="Recorded physical-regime partition")
            response_label=r"$f_D$"
            files.append(dataset)
        for workflow in local:
            scope=workflow["scope_id"];regime=int(scope.split("_")[-1])
            path=run/"tables"/(scope+"_collapse.csv");points=pd.read_csv(path);files.append(path)
            if not ((points.pi_1>0)&(points.response_transformed>0)).all():
                raise ValueError("Positive saved collapse coordinates are required for the showcase log axes")
            right.scatter(points.pi_1,points.response_transformed,s=11,color=colors[regime],alpha=.67,rasterized=True)
        right.set(xscale="log",yscale="log",xlabel=r"Selected local group $\Pi^*$",ylabel=response_label,title="Saved local dimensionless collapse")
        for axis in [left,right]:axis.grid(True,which="major",linewidth=.65);axis.set_axisbelow(True)
        handles=[Line2D([],[],marker="o",linestyle="",color=colors[r],markersize=6,label=label+f" (n={int((gradient.regime==r).sum())})") for r,label in names.items()]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.52,.075),ncol=len(names),frameon=False,fontsize=12)
        fig.text(.075,.025,"Each color shows a physical regime with its own saved local dimensionless group.",fontsize=11.5,color="#5A737E")
        save(fig,case_id.replace("_channel", "")+"_regimes.png",files,"Selected-run partition and pointwise local collapse; no surrogate, clustering or optimization is repeated.")

    # Same 83/21 validation split, redrawn without a dense source-symbol legend.
    folder=ROOT/"results/sandgrain_prediction";points=pd.read_csv(folder/"predictions.csv")
    fig=base_figure("Compact surface statistics, improved prediction", "Equivalent sand-grain roughness · identical 83 training and 21 test surfaces")
    for pos,column,label in [(.07,"original_seven","Seven original statistics"),(.59,"learned_groups","Two dimensionless groups")]:
        axis=fig.add_axes([pos,.30,.37,.48]);lo=0.;hi=25.5
        for split,color,face in [("train","#A4B4BE","white"),("test",TEAL,TEAL)]:
            part=points[points.split==split]
            axis.scatter(part.reference,part[column],s=38,facecolor=face,edgecolor=color,linewidth=1.25,zorder=3)
        axis.plot([lo,hi],[lo,hi],color=INK,linewidth=1.3)
        for factor in [.65,1.35]:axis.plot([lo,hi],[factor*lo,factor*hi],color="#93A9B4",linestyle=":",linewidth=1.3)
        test=points[points.split=="test"]
        r2=1-np.sum((test.reference-test[column])**2)/np.sum((test.reference-test.reference.mean())**2)
        axis.text(.055,.93,f"Conditional test R² = {r2:.3f}",transform=axis.transAxes,fontsize=12.5,color=BLUE)
        axis.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r"Reference $k_s/R_a$",ylabel=r"Predicted $k_s/R_a$",title=label)
        axis.grid(True,linewidth=.65);axis.set_axisbelow(True)
    fig.legend(handles=[Line2D([],[],marker="o",linestyle="",markerfacecolor="white",markeredgecolor="#A4B4BE",label="Training surfaces"),Line2D([],[],marker="o",linestyle="",color=TEAL,label="Test surfaces"),Line2D([],[],linestyle=":",color="#93A9B4",label="±35% relative-error bounds")],loc="lower center",bbox_to_anchor=(.52,.065),ncol=3,frameon=False,fontsize=12)
    fig.text(.07,.025,"Conditional validation uses the same saved training and test split for both representations.",fontsize=11.5,color="#5A737E")
    save(fig,"sandgrain_predictions.png",[folder/"predictions.csv",folder/"manifest.json"],"Direct redraw of the recorded source-stratified 83/21 prediction comparison; point values and split membership are unchanged.")

    # Fixed, archived Type II transformation; full profiles retain native order.
    folder=ROOT/"reproduction/velocity_comparison/results"
    profiles=pd.read_csv(folder/"transformed_native_profiles.csv");reference=pd.read_csv(folder/"incompressible_reference.csv")
    fig=base_figure("Recovering similarity with output rescaling", "Compressible wall turbulence · six archived DNS profiles · Type II representation")
    palette=["#245B78","#209B88","#C17837","#8A6BA3","#D25264","#5C8FA0"]
    axes=[fig.add_axes([.075,.30,.365,.48]),fig.add_axes([.60,.30,.365,.48])]
    for axis,xcol,ycol,title,xlabel,ylabel in [(axes[0],"y_plus","U_plus","Conventional wall units",r"$y^+$",r"$U^+$"),(axes[1],"x_AI","U_AI","Learned Type II rescaling",r"$\Pi_1^*$",r"$\widetilde{\Pi}_y$")]:
        for i,case in enumerate(sorted(profiles.case.unique())):
            part=profiles[(profiles.case==case)&(profiles[xcol]>0)]
            axis.plot(part[xcol],part[ycol],color=palette[i],linewidth=1.5)
        ref=reference[reference.y_plus>0]
        axis.plot(ref.y_plus,ref.U_plus,color=INK,linestyle="--",linewidth=1.8)
        axis.set(xscale="log",xlabel=xlabel,ylabel=ylabel,title=title)
        axis.grid(True,which="major",linewidth=.65);axis.set_axisbelow(True)
    handles=[Line2D([],[],color=palette[i],label=f"Profile {i+1}") for i in range(6)]+[Line2D([],[],color=INK,linestyle="--",label="Incompressible reference")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.53,.065),ncol=4,frameon=False,fontsize=12)
    fig.text(.075,.018,"Historical comparison data; zero wall-distance samples are omitted only from logarithmic display.",fontsize=11.5,color="#5A737E")
    save(fig,"velocity_transformations.png",[folder/"transformed_native_profiles.csv",folder/"incompressible_reference.csv"],"The archived algebraic AI Buckingham transformation is displayed without any fresh discovery, interpolation or fitting.")

    manifest={"schema_version":1,"generator":"scripts/build_showcase_assets.py","generator_sha256":sha256(Path(__file__)),
              "policy":"Original branding, an authorized figure-only workflow crop, and plotting of included numerical exports. No complete manuscript is distributed; no scientific source or result is modified.",
              "preferred_workflow_asset":"workflow.png",
              "png_canvas":{"width":1200,"height":550,"dpi":100,"main_font_points":15,"maximum_panels":2,"applies_to":[entry["file"] for entry in records if entry["file"].endswith(".png") and entry["file"]!="workflow.png"]},
              "attribution":"Project-created showcase graphics. Underlying datasets and third-party materials retain the terms documented in THIRD_PARTY_NOTICES.md.",
              "assets":records}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"docs/assets")
    parser.add_argument("--workflow-pdf", type=Path, help="Optional local author manuscript used only to render Fig. 2; it is never copied into the output.")
    parser.add_argument("--workflow-only", action="store_true", help="Update/verify only workflow.png and its manifest entry; leave all other assets unchanged.")
    parser.add_argument("--workflow-dpi", type=int, default=400)
    args=parser.parse_args()
    if args.workflow_dpi < 72:
        parser.error("--workflow-dpi must be at least 72")
    if args.workflow_only:
        manifest=update_workflow_only(args.output_dir.resolve(), args.workflow_pdf, args.workflow_dpi)
    else:
        manifest=build(args.output_dir.resolve(), args.workflow_pdf, args.workflow_dpi)
    print(json.dumps({"assets":len(manifest["assets"]),"output":str(args.output_dir.resolve())}))


if __name__=="__main__":
    main()
