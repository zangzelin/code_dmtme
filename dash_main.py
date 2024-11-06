import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from dashfile.data_processing import load_data_visemb_from_lightning_ckpt, plot_scatter_hyper
import os

current_figs = [None, None]


app = dash.Dash(__name__)

datas, embeddings, exp_emb_list, ins_exp_mask, fea_ins_mask, xmask, labels = load_data_visemb_from_lightning_ckpt(
    'zzl_checkpoints_exp/HCL/best_model_HCL_acc0.8371000000000001.pth', dataset='HCL60KPLOT', T_num_layers=10, sample_rate_feature=0.50, num_input_dim=3038,
    sample_num=60000)

image_files = [f'path/to/your/images/image{i}.png' for i in range(1, 101)]  # Replace with actual paths to your images
# import pdb; pdb.set_trace()
scatter_fig, heatmap_fig, secondary_heatmap_fig, encoded_images, encoded_images_exp = plot_scatter_hyper(
    datas, 
    embeddings, 
    labels, 
    exp_emb=exp_emb_list,
    R=1.5, 
    image_files=image_files,
    xmask_all=xmask,
    fea_ins_mask=fea_ins_mask[:,:2000],
)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='scatter-plot', figure=scatter_fig)
    ], style={'width': '70%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='heatmap1', figure=heatmap_fig),
    ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
    dcc.Store(id='encoded-images-store', data=encoded_images),  # 存储 encoded_images 数据
    dcc.Store(id='encoded_images_exp-store', data=encoded_images_exp)  # 存储 encoded_images_exp 数据
])

@app.callback(
    Output('heatmap1', 'figure'),
    Input('scatter-plot', 'clickData'),
    Input('encoded-images-store', 'data'),
)
def update_heatmap1(clickData, encoded_images):
    global current_figs

    if clickData is None:
        raise dash.exceptions.PreventUpdate

    point_index = clickData['points'][0]['customdata']  # 获取点击点的索引
    # print(clickData['points'][0]['text'], "clickData['points'][0]['text']")
    # selected_image = encoded_images[point_index]  # 获取对应的图像数据

    if 'Label' in clickData['points'][0]['text']:
        heatmap_fig = go.Figure(data=go.Heatmap(z=point_index, colorscale='Greys'))
        # current_figs[0] = heatmap_fig
        # print('Label ------------')
        # return current_figs
        
    if 'Exp' in clickData['points'][0]['text']:
        heatmap_fig = go.Figure(data=go.Heatmap(z=point_index, colorscale='Magma'))
        # current_figs[1] = heatmap_fig
        # current_figs[0] = heatmap_fig
        # print('Label ************')
        # return heatmap_fig, heatmap_fig
    return heatmap_fig
        

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)