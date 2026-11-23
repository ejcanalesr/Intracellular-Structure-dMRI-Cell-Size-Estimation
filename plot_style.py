"""Shared paper colors, final method names and fonts. Edit here for all figures."""
from pathlib import Path
import hashlib
STYLE_VERSION='2026-11-23'

MODEL_COLORS={
    'M0-expm-truncated':'#2878A2',
    'M0-Strang':'#D57824',
    'M0-GPA':'#81539D',
}

MODEL_LABELS={
    'M0-expm-truncated':'M0 (150 modes)',
    'M0-Strang':'M0-Strang',
    'M0-GPA':'M0-GPA',
}

PROTOCOL_LABELS={'P1':'P1: SANDI Connectome 2.0','P2':'P2: POLAR-CEXI'}

PROTOCOL_ORDER=['P1','P2']

MODEL_ORDER=['M0-expm-truncated','M0-Strang','M0-GPA']

PROTOCOL_ONLY_COLORS={'P1':'#2878A2','P2':'#D57824'}

PROTOCOL_ONLY_MARKERS={'P1':'o','P2':'^'}

FONT_SIZE=8.5

FONT_FAMILY='DejaVu Sans'

MATH_FONTSET='dejavusans'

TITLE_FONT_SIZE=9.3

PANEL_FONT_SIZE=10.5

LEGEND_FONT_SIZE=8.3

FIGURE_WIDTH_INCHES=7.6

JOINT_FIGURE_WIDTH_INCHES=8.0

BOX_OFFSET=.08

BOX_WIDTH=.32

BOX_FACE_ALPHA=.20

MEDIAN_LINEWIDTH=1.15

WHISKER_LINEWIDTH=.7

WHISKER_IQR=1.5

GRID_ALPHA=.13

SAMPLE_POINTS_ALPHA=.30

MEAN_MARKER='D'

TRUTH_MARKER='*'

TRUTH_LINESTYLE=':'

PNG_DPI=300

EXPORT_FORMATS=['pdf','svg','png']

DIFFUSIVITY_STYLES={2.0:{'linestyle':'-','marker':'o'},.6:{'linestyle':'--','marker':'s'}}

AXIS_LABELS={
    'true_R':r'True $R$ ($\mu$m)',
    'estimated_R':r'Estimated $R$ ($\mu$m)',
    'estimated_D':r'Estimated $D$ ($\mu$m$^2$/ms)',
    'radius_error_percent':'Radius error (%)',
    'max_signal_error':r'Max. signal error ($S_0$)',
    'permeability':r'$\kappa$ ($\mu$m/s)',
    'diffusivity_ratio':r'$D_n/D_c$',
}


def settings():
    """A fresh simple dictionary; changing it cannot mutate the shared defaults."""
    return dict(style_version=STYLE_VERSION,model_colors=MODEL_COLORS.copy(),
        method_labels=MODEL_LABELS.copy(),protocol_labels=PROTOCOL_LABELS.copy(),
        protocol_columns=PROTOCOL_ORDER.copy(),font_size=FONT_SIZE,font_family=FONT_FAMILY,
        math_fontset=MATH_FONTSET,title_font_size=TITLE_FONT_SIZE,legend_font_size=LEGEND_FONT_SIZE,
        panel_font_size=PANEL_FONT_SIZE,width_inches=FIGURE_WIDTH_INCHES,
        box_offset_categories=BOX_OFFSET,box_width_categories=BOX_WIDTH,box_face_alpha=BOX_FACE_ALPHA,
        median_linewidth=MEDIAN_LINEWIDTH,whisker_linewidth=WHISKER_LINEWIDTH,
        whisker_IQR=WHISKER_IQR,grid_alpha=GRID_ALPHA,pdf_fonttype=42,dpi=PNG_DPI)


def apply_style():
    """Set the established matplotlib defaults; creates no figure or data."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':FONT_SIZE,'font.family':FONT_FAMILY,'mathtext.fontset':MATH_FONTSET,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})


def save_pdf(fig,filename):
    filename=Path(filename).with_suffix('.pdf')
    filename.parent.mkdir(parents=True,exist_ok=True)
    style_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fig.savefig(filename,dpi=300,metadata={'Subject':'Panel labels: uppercase bold, no parentheses',
        'Keywords':'plot_style.py SHA256='+style_hash})


def panel_label(ax,index):
    """Shared subplot label: A, B, C...; bold, upper left, no parentheses.

    index is zero-based in row-major panel order. text2D keeps labels in the
    same axes coordinates for both 3D mesh views and ordinary 2D plots.
    """
    text=ax.text2D if hasattr(ax,'text2D') else ax.text
    return text(-.13,1.04,chr(65+index),transform=ax.transAxes,
        fontsize=PANEL_FONT_SIZE,fontweight='bold',color='black',
        ha='left',va='bottom',clip_on=False)
