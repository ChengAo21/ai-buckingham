from html import escape
import base64
from pathlib import Path
import streamlit as st
from ..catalog import ROOT

CSS='''
.stApp{background:#f7f6f1;color:#243c42} [data-testid="stToolbar"]{background:transparent} [data-testid="stAppDeployButton"]{display:none} [data-testid="stHeader"]{background:transparent} [data-testid="stSidebar"]{background:#eaece5;border-right:1px solid #d9dfd5}
[data-testid="stSidebar"]:has(button[aria-label="Collapse sidebar"][aria-expanded="true"]){width:224px!important;min-width:224px!important;max-width:224px!important}
.block-container{max-width:1260px;padding:3.4rem 2.5rem 3rem}h1,h2,h3{letter-spacing:-.035em;color:#243c42}
.stApp h1{font:600 2.5rem/1.14 Georgia,serif!important}.stApp h2{font:600 1.55rem/1.2 Georgia,serif!important}.stApp h3{font-size:1.12rem!important}
[data-testid="stSidebar"] .block-container{padding-top:2rem}.stButton>button,.stDownloadButton>button{border-radius:7px;font-size:.85rem}
[data-testid="stTabs"] [role="tablist"]{gap:22px;border-bottom:1px solid #d6ddd4;margin-bottom:24px}[role="tab"]{font-size:.9rem!important}
[data-testid="stMetric"]{padding:14px 18px;background:#fff;border:1px solid #e0e4dc;border-radius:8px}[data-testid="stMetricValue"]{font-size:1.6rem!important}
.eyebrow{font:700 .68rem Arial,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#548377;margin:0 0 12px}.intro{color:#6c7976;font-size:.94rem;line-height:1.6;margin:8px 0 26px}
.hero{position:relative;overflow:hidden;border-radius:14px;background:#203f43;color:#f7f6f1;padding:32px 36px;margin-bottom:28px;border:1px solid #305356}
.hero:after{content:none}.hero-copy{max-width:calc(100% - 165px)}.hero-robot{position:absolute;right:26px;top:50%;transform:translateY(-50%);width:145px;height:145px;object-fit:contain}.hero-capabilities{display:flex;gap:18px;flex-wrap:wrap;margin-top:18px}.hero-capabilities>div{display:flex;gap:10px;align-items:center}.hero-capabilities>div>span{border:1px solid #80ab9b;border-radius:7px;width:30px;height:32px;text-align:center;font:22px Georgia,serif;padding-top:3px;color:#c7e4d9}.hero-capabilities b{font:600 .83rem Arial,sans-serif;color:#e5f0e8}.hero-capabilities small{display:block;color:#adc9bb;font-size:.7rem;font-weight:400;margin-top:5px}
.hero .eyebrow{color:#a7cebc}.hero h1{color:#fff!important;font-size:3rem!important;margin:0 0 12px;padding:0!important}.hero p{position:relative;z-index:1;max-width:620px;color:#d0dfd7;font-size:.97rem;line-height:1.6;margin-bottom:20px}
.hero-facts{display:flex;gap:30px;font-size:.76rem;color:#b7cec3;position:relative;z-index:1}.hero-facts b{font:22px Georgia,serif;color:#fff;margin-right:7px}
.section-title{display:flex;justify-content:space-between;align-items:baseline;margin:26px 0 14px;gap:20px}.section-title h2{margin:0!important;padding:0!important}.section-title p{font-size:.79rem;color:#73817a;margin:0}
.scheme{background:#fff;border:1px solid #dce1d7;border-radius:12px;overflow-x:auto;margin-bottom:26px}.scheme img{width:100%;min-width:760px;height:auto;display:block}
.case-heading{font:600 .96rem/1.25 Georgia,serif;min-height:2.5em;margin-bottom:8px;color:#263e44}.case-meta{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:.73rem;color:#6b7973;margin:10px 0}.tag{font:600 .67rem Arial,sans-serif;padding:4px 8px;border-radius:20px;background:#e5eee8;color:#336c60}.tag.type-ii{background:#eee8f2;color:#79618e}.tag.multi-regime{background:#f5ebdb;color:#9a713c}
[class*="st-key-case_card_"]{background:#fff;border:1px solid #dce1d7!important;border-radius:12px!important;padding:13px!important}
[class*="st-key-case_card_"] button{background:#eff3ed;color:#345a51;border-color:#d5dfd3}[class*="st-key-case_card_"] [data-testid="stImage"]{background:white}
.case-banner{display:grid;grid-template-columns:1.3fr 1fr;gap:25px;align-items:center;background:#fff;border:1px solid #dce1d7;border-radius:12px;padding:16px 24px;margin:8px 0 18px}.case-banner h1{font-size:1.8rem!important;margin:6px 0!important;padding:0!important}.case-banner p{font-size:.84rem;color:#6b7973;margin:8px 0}.case-banner img{width:100%;height:110px;object-fit:contain}
.formula{background:#edf1e9;border-left:3px solid #5e8d77;border-radius:5px;padding:17px 22px;font:1.13rem/1.5 Georgia,serif;margin:10px 0 20px;color:#345b50}
.criteria{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0 22px}.criterion{border:1px solid #d7dfd2;border-radius:9px;padding:15px 16px;background:linear-gradient(135deg,#eef3e9,#fff)}.criterion b{display:block;font:27px Georgia,serif;color:#35665b;margin-bottom:5px}.criterion span{font-size:.74rem;color:#69796f}.criterion small{display:block;font-size:.66rem;color:#8a958c;margin-top:4px}
.run-step{font:700 .7rem Arial,sans-serif;color:#4d7b68;letter-spacing:.12em;text-transform:uppercase;border-top:1px solid #d6ddd2;padding-top:18px;margin:18px 0 10px}.sidebar-mark{font:28px Georgia,serif;color:#2c514b;padding:4px 0 8px}.sidebar-note{font-size:.72rem;color:#758578;line-height:1.7;margin-top:20px}.plot-title{font:600 .84rem Arial,sans-serif;color:#49625c;margin:14px 0 0}
.sidebar-brand{display:flex;gap:12px;align-items:center;margin:8px 0 28px}.sidebar-brand>span{font:34px Georgia,serif;background:#294d48;color:#e7f0e4;width:42px;height:46px;text-align:center;border-radius:7px}.sidebar-brand strong{font:600 19px/1.15 Georgia,serif;color:#294d48}footer{visibility:hidden}
@media(max-width:1120px){[data-testid="stHorizontalBlock"]:has(>[data-testid="stColumn"] [data-testid="stPlotlyChart"]){flex-wrap:wrap}[data-testid="stHorizontalBlock"]:has(>[data-testid="stColumn"] [data-testid="stPlotlyChart"])>[data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}}
@media(max-width:800px){.block-container{padding:3rem 1.2rem}.case-banner{grid-template-columns:1fr;gap:5px}.criteria{grid-template-columns:repeat(2,1fr)}.hero{padding:25px}.hero h1{font-size:2.5rem!important}.hero-facts{gap:17px;flex-wrap:wrap}.hero:after{right:-40px}.section-title p{display:none}}
'''

@st.cache_data(show_spinner=False)
def uri(path,revision=0):
    p=Path(path);kind='svg+xml' if p.suffix=='.svg' else 'png'
    return f'data:image/{kind};base64,'+base64.b64encode(p.read_bytes()).decode()

def image(path):return uri(str(ROOT/path),(ROOT/path).stat().st_mtime_ns)
def heading(title,detail=None):
    st.markdown(f'<div class="section-title"><h2>{escape(title)}</h2>'+ (f'<p>{escape(detail)}</p>' if detail else '')+'</div>',unsafe_allow_html=True)
def intro(title,text):
    st.markdown(f'<h1>{escape(title)}</h1><p class="intro">{escape(text)}</p>',unsafe_allow_html=True)
def formula(text):st.markdown(f'<div class="formula">{escape(text)}</div>',unsafe_allow_html=True)
def plot_title(text):st.markdown(f'<div class="plot-title">{escape(text)}</div>',unsafe_allow_html=True)

CSS += '[class*="st-key-case_gallery_"]{max-width:1080px}.scheme{padding:4px}.stApp [data-testid="stLatex"]{overflow-x:auto}@media(max-width:800px){.hero-copy{max-width:100%}.hero-robot{display:none}.hero-capabilities{gap:16px}.hero p{font-size:.92rem}}'

CSS += '''
.stApp{font-family:"Source Sans 3","Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.55}
.stApp h1,.stApp h2{font-family:Georgia,"Times New Roman",serif!important;letter-spacing:-.025em}
.stApp h2{font-size:1.65rem!important}.stApp h3{font-size:1.16rem!important;letter-spacing:-.015em}
.stApp [data-testid="stCaptionContainer"]{font-size:.84rem;line-height:1.5;color:#6b7e75}
.stApp [data-testid="stWidgetLabel"] p{font-size:.9rem;font-weight:500}
[role="tab"]{font-size:.95rem!important;font-weight:600}
.plot-title{font-size:.94rem;font-weight:600;line-height:1.35;margin:12px 0 10px;color:#365750}
.section-title{margin:24px 0 14px}.section-title p{font-size:.84rem}
[class*="st-key-case_gallery_"]{max-width:none!important}
[data-testid="stLayoutWrapper"]:has(>[class*="st-key-case_card_"]){flex:1 1 270px!important;width:auto!important;min-width:min(100%,250px)!important;max-width:none!important}
[class*="st-key-case_card_"]{flex:1 1 260px!important;width:auto!important;min-width:min(100%,250px)!important;max-width:none!important}
[class*="st-key-case_card_"] [data-testid="stImage"] img{height:145px;object-fit:contain}
.case-heading{font-size:1.12rem;line-height:1.3;min-height:2.6em;margin-bottom:5px}
[class*="st-key-scaling_formula_"]{background:#edf3ee;border-color:#d9e5db!important;padding:8px 16px!important;margin:8px 0 14px}
[class*="st-key-scaling_formula_"] [data-testid="stLatex"]{margin:0;padding:4px 0}
[class*="st-key-search_controls_"]{background:#fff;border-color:#dce5dc!important;padding:8px 16px 14px!important}
[class*="st-key-search_controls_"] .plot-title{margin-top:2px;margin-bottom:12px}
[class*="st-key-learning_plot_"]{flex:1 1 320px!important;width:auto!important;min-width:min(100%,300px)!important}
[data-testid="stLayoutWrapper"]:has(>[class*="st-key-learning_plot_"]){flex:1 1 320px!important;width:auto!important;min-width:min(100%,300px)!important}
.optimization-key{display:flex;justify-content:center;flex-wrap:wrap;gap:12px 26px;padding:8px 12px 13px;margin:0 0 8px;border-bottom:1px solid #dce4dc;font-size:.87rem;color:#36524f}
.optimization-key>span{display:inline-flex;align-items:center;gap:9px}
.optimization-key i{display:inline-block;flex:none}.key-line{width:24px;height:3px;background:#e63946}.key-best{width:24px;height:3px;background:#507e74}
.key-start{width:9px;height:9px;border:1.2px solid #19344f;border-radius:50%;background:white}.key-end{width:9px;height:9px;border:1.2px solid #19344f;background:#f4c95d;transform:rotate(45deg)}
[class*="st-key-eigen_matrix_"]{flex:1 1 250px!important;min-width:0!important}
[data-testid="stLayoutWrapper"]:has(>[class*="st-key-eigen_matrix_"]){flex:1 1 250px!important;width:auto!important;min-width:0!important}
@media(max-width:900px){[class*="st-key-search_controls_"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap}[class*="st-key-search_controls_"] [data-testid="stColumn"]{min-width:135px;flex:1 1 135px!important}}
@media(max-width:600px){.block-container{padding:2.4rem 1.15rem 2rem}.hero h1{font-size:2.25rem!important}.criteria{gap:7px}.criterion{padding:12px}.case-heading{min-height:0}[class*="st-key-case_card_"]{min-width:100%!important}}
'''
