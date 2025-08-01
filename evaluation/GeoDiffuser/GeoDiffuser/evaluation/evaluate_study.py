import pandas as pd
import argparse
import numpy as np
import plotly.express as px

def read_csv(csv_file):

    df = pd.read_csv(csv_file)
    return df

def convert_pd_to_numpy(pd_df, arr_type=None):

    pd_np = pd_df.to_numpy()

    if arr_type is not None:
        pd_np = pd_np.astype(arr_type)

    return pd_np


def draw_graph(data, data_total, mask):


    # print(px.data.medals_long())
    p  = np.sum(data * mask) / np.sum(data_total * mask)
    
    data_y = np.array([p * 100, (1.0 - p) * 100])

    data_x = np.array(["Ours", "Lama"])

    fig = px.bar(data_x, y = data_y)
    fig.write_image("test/fig1.png")
    pass

def get_percentage(t, t_total, mask):
    p  = np.sum(t * mask) / np.sum(t_total * mask) * 100.0
    return p

def draw_all_graphs(transforms_realism, transforms_realism_total, transforms_edit_adherance, transforms_edit_adherance_total, removal, removal_total, survey_ip_mask):

    p_er = get_percentage(transforms_realism, transforms_realism_total, survey_ip_mask)
    p_ea = get_percentage(transforms_edit_adherance, transforms_edit_adherance_total, survey_ip_mask)
    p_removal = get_percentage(removal, removal_total, survey_ip_mask)
    
    print(np.sum(survey_ip_mask))

    # print(p_ea)
    d = {"Study Type": {0: "Edit Realism", 1: "Edit Adherance", 2: "Removal Realism", 3: "Edit Realism", 4: "Edit Adherance", 5: "Removal Realism"},   
         "Method": {0: "GeoDiffuser (Ours)", 1: "GeoDiffuser (Ours)", 2:"GeoDiffuser (Ours)", 3: "Zero123XL + Lama", 4: "Zero123XL + Lama", 5: "Lama"},
         "Participant Preference %": {0: p_er, 1: p_ea, 2: p_removal, 3: 100.0 - p_er, 4: 100.0 - p_ea, 5: 100.0 - p_removal}}


    df = pd.DataFrame.from_dict(d)
    fig = px.bar(df, x="Study Type", y="Participant Preference %", color="Method", text_auto='.2f')

    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ),
    font=dict(
        # family="Courier New, monospace",
        size=19,  # Set the font size here
        # color="RebeccaPurple"
    ))
    fig.update_traces(text= [f'{val}\u00A3' for val in df['Participant Preference %']])
    fig.write_image("test/study_graph.jpg", scale=6, )#width=1000, height=500)
    
    
    pass
if __name__ == "__main__":


    start_idx = 2
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--csv_file',
        required=True, default = None)
    args = parser.parse_args()

    df = read_csv(args.csv_file)

    survey_type = convert_pd_to_numpy(df["Status"].loc[start_idx:])
    # print(survey_type != "Survey Preview")


    survey_ip_mask = (survey_type != "Survey Preview") * 1.0
    transforms_realism = convert_pd_to_numpy(df["SC1"].loc[start_idx:], "int")
    transforms_realism_total = convert_pd_to_numpy(df["SC4"].loc[start_idx:], "int")

    transforms_edit_adherance = convert_pd_to_numpy(df["SC2"].loc[start_idx:], "int")
    transforms_edit_adherance_total = convert_pd_to_numpy(df["SC5"].loc[start_idx:], "int")

    removal = convert_pd_to_numpy(df["SC3"].loc[start_idx:], "int")
    removal_total = convert_pd_to_numpy(df["SC6"].loc[start_idx:], "int")



    print("Removal: ", np.sum(removal * survey_ip_mask) / np.sum(removal_total* survey_ip_mask))
    print("Realism: ",np.sum(transforms_realism* survey_ip_mask) / np.sum(transforms_realism_total* survey_ip_mask))
    print("Adherance: ",np.sum(transforms_edit_adherance* survey_ip_mask) / np.sum(transforms_edit_adherance_total* survey_ip_mask))


    # draw_graph(removal, removal_total, survey_ip_mask)

    draw_all_graphs(transforms_realism, transforms_realism_total, transforms_edit_adherance, transforms_edit_adherance_total, removal, removal_total, survey_ip_mask)
    # print(transforms_realism)
    # print(removal / removal_total)
    # print(df["SC1"])


    # print(df.loc[0]["SC1"])

    pass