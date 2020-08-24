'''
deep-racer sim-trace-log analizer
'''
import os
import sys
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1


def file2df(filepath):
    with open(filepath, "r") as log_file:
        lines = log_file.readlines()
    log_lines = []

    # SIM_TRACE_LOGを抽出
    for line in lines:
        if line.startswith("SIM_TRACE_LOG:"):
            log_lines.append(
                list(line[line.find(":")+1:line.find("/n")].split(",")))

    # metrics取得
        if "METRICS_S3_BUCKET" in line:
            metrics_str = line[line.find("{"):line.find("}")+1]
    #         print(metrics_str)
            if metrics_str:
                metrics = ast.literal_eval(metrics_str)

    # model名取得
    key = str(metrics['METRICS_S3_OBJECT_KEY'])
    model_name = key[key.find('models')+7:key.find('metrics')-1]
    print("ModelName:%s" % model_name)

    # 列名指定
    log_col = ['episode', 'step', 'x-coordinate', 'y-coordinate', 'heading', 'steering_angle', 'speed', 'action_taken', 'reward',
               'job_completed', 'all_wheels_on_track', 'progress', 'closest_waypoint_index', 'track_length', 'timestamp', 'status']

    log_df = pd.DataFrame(data=log_lines, columns=log_col)

    # 型指定
    log_df = log_df.astype({'episode': int, 'step': int, 'x-coordinate': float, 'y-coordinate': float, 'heading': float, 'steering_angle': float, 'speed': float, 'action_taken': int, 'reward': float,
                            'job_completed': bool, 'all_wheels_on_track': bool, 'progress': float,  'closest_waypoint_index': int, 'track_length': float, 'timestamp': str, 'status': str})

    return log_df, metrics


def summary_episode(log_df):

    sum_col = ['episode', 'steps', 'rewards', 'completed', 'laptime', 'status']
    sum_df = pd.DataFrame(columns=sum_col)

    for i in range(int(log_df['episode'].max()+1)):

        log_df_ep = log_df[log_df['episode'] == i]

        episode = log_df_ep['episode'].min()
        total_steps = log_df_ep['step'].max()
        total_rewards = round(log_df_ep['reward'].sum(), 3)
        track_completed = round(log_df_ep['progress'].max(), 2)
        finish_time = float(log_df_ep['timestamp'].max())
        start_time = float(log_df_ep['timestamp'].min())

        laptime = round(finish_time - start_time, 3)

        status = log_df_ep['status'].tail(1).values
        sum_se = pd.Series([episode, total_steps, total_rewards,
                            track_completed, laptime, status], index=sum_col)
        sum_df = sum_df.append(sum_se, ignore_index=True)
    return sum_df


def select_top5(sum_df):
    top_df = sum_df.sort_values(
        ['completed', 'laptime'], ascending=[False, True]).head(5)
    return top_df


def plot_ax(fig, ax, col, cmap, title, top_df, log_df, track):
    #  コース描画
    ax.plot(track[:, 0],  track[:, 1], c="grey", alpha=0.5, linestyle=':')
    ax.plot(track[:, 2],  track[:, 3], c="brown")
    ax.plot(track[:, 4],  track[:, 5], c="brown")

    # #  走行ライン描画
    for i, epi_se in top_df.iterrows():
        epi_df = log_df.query('episode==%i' % epi_se['episode'])
        epi_ar = epi_df.values
        ax.plot(epi_ar[:, 2], epi_ar[:, 3], c="grey", alpha=0.3)

    # top5エピソードのログ抽出
    epi_no = list(top_df['episode'].values)
    top_five_ar = log_df[log_df['episode'].isin(epi_no)].values

    # 点描画
    colormap = plt.get_cmap(cmap)
    mapa = ax.scatter(top_five_ar[:, 2], top_five_ar[:, 3],
                      c=top_five_ar[:, col], marker='.', cmap=colormap)

    # カラーバー調整
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    fig.colorbar(mapa, cax=cax)

    # 全体調整
    ax.set_title(title)
    ax.set_aspect('equal', 'box')


def read_track(metrics):
    # トラック読み込み
    print("WorldName:%s" % metrics['WORLD_NAME'])
    WORLD_NAME = metrics['WORLD_NAME']
    track = np.load("%s/tracks/%s.npy" % (os.getcwd(), WORLD_NAME))
    return track


def save_top5_fig(top_df, log_df, metrics):
    WORLD_NAME = metrics['WORLD_NAME']
    track = np.load("%s/tracks/%s.npy" % (os.getcwd(), WORLD_NAME))

    # model名取得
    key = str(metrics['METRICS_S3_OBJECT_KEY'])
    model_name = key[key.find('models')+7:key.find('metrics')-1]

# グラフ作成
    fig = plt.figure(figsize=(20, 20), dpi=200, facecolor='w')
    # Top5 Speed
    ax = fig.add_subplot(2, 2, 1)
    plot_ax(fig, ax, 6, 'jet', 'Speed', top_df, log_df, track)
    # Top5 Steering
    ax = fig.add_subplot(2, 2, 2)
    plot_ax(fig, ax, 5, 'jet', 'Steering Angle', top_df, log_df, track)
    # Top5 Reward
    ax = fig.add_subplot(2, 2, 3)
    plot_ax(fig, ax, 8, 'jet', 'Reward', top_df, log_df, track)
    # Top5 summary
    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=top_df.values,
             colLabels=top_df.columns,
             loc='center')

    fig.tight_layout()
    fig.suptitle('%s Top 5 Episode' % model_name)

    fig.show()
    fig.savefig('%s/img/%s_top5.png' % (os.getcwd(), model_name))


def main():
    # get parameters
    filepath = ''
    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            filepath = sys.argv[1]
        else:
            print('robomaker file not exist')
            return

    log_df, metrics = file2df(filepath)
    top_df = select_top5(summary_episode(log_df))
    save_top5_fig(top_df, log_df, metrics)


if __name__ == '__main__':
    main()
