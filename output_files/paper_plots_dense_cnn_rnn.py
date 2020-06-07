import numpy as np
import json
import warnings
import operator

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

base_path = 'data_20_05/'

all_approaches_path = ['dnn/', 'dnn_wc/', 'cnn/', 'cnn_wc/', 'gru/', 'gru_wc/']

titles = ['(a) Dense neural network (DNN)', '(b) DNN with weighted loss', '(c) Convolutional neural network (CNN)', '(d) CNN with weighted loss', '(e) Recurrent neural network (GRU)', '(f) GRU with weighted loss']

font = {'family': 'serif', 'size': 24}

alpha_fade = 0.1
fig_size = (12, 12)

plt.rc('font', **font)

size_title = 28
size_label = 24
runs = 10
epochs = 10

loss_ylim = (0.0, 1.0)
usage_ylim = (1.0, 5.0)
top_legend = ['Top1', 'Top2']

gs = gridspec.GridSpec(3,2)
leg_loc = 4
leg_size = 18


def read_file(path):
    with open(path) as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = list(map(float, data))
        return data


def extract_precision(precision_path):

    top1_compatible_precision = list()
    top2_compatible_precision = list()
    top3_compatible_precision = list()
    with open(precision_path) as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = data[:epochs]
        for row in data:
            row = row.split('\n')
            row = row[0].split(' ')
            row = list(map(float, row))
            top1_compatible_precision.append(row[0])
            top2_compatible_precision.append(row[1])
            top3_compatible_precision.append(row[2])
    return top1_compatible_precision, top2_compatible_precision, top3_compatible_precision


def compute_fill_between(a_list):
    y1 = list()
    y2 = list()
    a_list = np.array(a_list, dtype=float)
    n_cols = a_list.shape[1]
    for i in range(0, n_cols):
        pos = a_list[:, i]
        std = np.std(pos)
        y1.append(std)
        y2.append(std)
    return y1, y2


def plot_loss(ax, x_val1, loss_tr_y1, loss_tr_y2, x_val2, loss_te_y1, loss_te_y2, title, xlabel, ylabel, leg):
    x_val1 = x_val1[:epochs]
    x_val2 = x_val2[:epochs]
    loss_tr_y1 = loss_tr_y1[:epochs]
    loss_tr_y2 = loss_tr_y2[:epochs]
    loss_te_y1 = loss_te_y1[:epochs]
    loss_te_y2 = loss_te_y2[:epochs]
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, loss_tr_y1, loss_tr_y2, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, loss_te_y1, loss_te_y2, color = 'b', alpha = alpha_fade)
    ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(loss_ylim)
    ax.grid(True)


def assemble_loss():
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Cross-entropy loss for multiple neural network architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):            
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Loss", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Loss", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Loss", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            
        train_loss = list()
        test_loss = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            tr_loss_path = path + 'train_loss.txt'
            val_loss_path = path + 'validation_loss.txt'
            try:
                tr_loss = read_file(tr_loss_path)
                train_loss.append(tr_loss)
                te_loss = read_file(val_loss_path)
                test_loss.append(te_loss)
            except Exception:
                continue
        loss_tr_y1, loss_tr_y2 = compute_fill_between(train_loss)
        loss_te_y1, loss_te_y2 = compute_fill_between(test_loss)

        mean_tr_loss = np.mean(train_loss, axis=0)
        mean_te_loss = np.mean(test_loss, axis=0)
        plt_title = titles[idx]
        plot_loss(ax, mean_tr_loss, mean_tr_loss - loss_tr_y1, mean_tr_loss + loss_tr_y2, mean_te_loss, mean_te_loss - loss_te_y1, mean_te_loss + loss_te_y2, plt_title + "", "Training iterations (epochs)", "Mean loss", ['Training loss', 'Test (validation) loss'])
#assemble_loss()
plt.show()


def plot_usage(ax, x_val1, y1_top1, y2_top1, x_val2, y1_top2, y2_top2, x_val3, y1_top3, y2_top3, title, xlabel, ylabel, leg):
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    #ax.plot(x_pos, x_val3, 'g')
    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, y1_top1, y2_top1, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, y1_top2, y2_top2, color = 'b', alpha = alpha_fade)
    #ax.fill_between(x_pos, y1_top3, y2_top3, color = 'g', alpha = alpha_fade)
    ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(usage_ylim)
    ax.grid(True)

def assemble_usage():
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean log usage frequency for multiple neural network architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):        
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Log usage", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Log usage", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Log usage", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        usage_top1 = list()
        usage_top2 = list()
        usage_top3 = list()
        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            usage_path = path + 'usage_weights.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(usage_path)
                usage_top1.append(top1_p)
                usage_top2.append(top2_p)
                usage_top3.append(top3_p)
            except Exception:
                continue
        mean_top1_usage = np.mean(usage_top1, axis=0)
        mean_top2_usage = np.mean(usage_top2, axis=0)
        mean_top3_usage = np.mean(usage_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(usage_top1)
        y1_top2, y2_top2 = compute_fill_between(usage_top2)
        y1_top3, y2_top3 = compute_fill_between(usage_top3)
        plt_title = titles[idx]
        leg = top_legend
        plot_usage(ax, mean_top1_usage, mean_top1_usage - y1_top1, mean_top1_usage + y2_top1, mean_top2_usage, mean_top2_usage - y1_top2, mean_top2_usage + y2_top2, mean_top3_usage, mean_top3_usage - y1_top3, mean_top3_usage + y2_top3, plt_title, "Training iterations (epochs)", "Mean log usage frequency", leg)
assemble_usage()
plt.show()


def plot_accuracy(ax, x_val1, y1_top1, y2_top1, x_val2, y1_top2, y2_top2, x_val3, y1_top3, y2_top3, title, xlabel, ylabel, leg=top_legend, precision_ylim=(0.4, 1.2)):
    x_pos = np.arange(len(x_val1))
    ax.plot(x_pos, x_val1, 'r')
    ax.plot(x_pos, x_val2, 'b')
    #ax.plot(x_pos, x_val3, 'g')

    ax.set_title(title, size=size_title)
    ax.fill_between(x_pos, y1_top1, y2_top1, color = 'r', alpha = alpha_fade)
    ax.fill_between(x_pos, y1_top2, y2_top2, color = 'b', alpha = alpha_fade)
    #ax.fill_between(x_pos, y1_top3, y2_top3, color = 'g', alpha = alpha_fade)
    ax.legend(top_legend, loc=leg_loc, prop={'size': leg_size})
    ax.set_ylim(precision_ylim)
    plt.grid(True)

def assemble_accuracy(sup_title):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(sup_title, size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'precision.txt'
    
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k")
assemble_accuracy('Mean normal precision@k for multiple neural network architectures')
plt.show()


def assemble_published_precision(sup_title):
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(sup_title, size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'published_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k")
assemble_published_precision('Mean standard precision@k for multiple neural network architectures')
plt.show()


def assemble_lowest_normal_precision():
    precision_ylim = (0.25, 1.0)
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean precision@k in lowest 25% of data for multiple architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'lowest_norm_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k", ['Top1', 'Top2', 'Top3'], precision_ylim)
#assemble_lowest_normal_precision()
plt.show()


def assemble_lowest_published_precision():
    precision_ylim = (0.0, 0.4)
    fig = plt.figure(figsize=fig_size)
    fig.suptitle('Mean standard precision@k in lowest 25% of data for multiple architectures', size=size_title + 2)
    for idx, approach in enumerate(all_approaches_path):
        if idx == 0:
            ax = plt.subplot(gs[0,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 1:
            ax = plt.subplot(gs[0,1])
        elif idx == 2:
            ax = plt.subplot(gs[1,0])
            ax.set_ylabel("Precision@k", size=size_label)
        elif idx == 3:
            ax = plt.subplot(gs[1,1])
        elif idx == 4:
            ax = plt.subplot(gs[2,0])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)
            ax.set_ylabel("Precision@k", size=size_label)
        else:
            ax = plt.subplot(gs[2,1])
            ax.set_xlabel("Training iterations (epochs)", size=size_label)

        precision_acc_top1 = list()
        precision_acc_top2 = list()
        precision_acc_top3 = list()

        for i in range(1, runs+1):
            path = base_path + approach + 'run' + str(i) + '/'
            precision_path = path + 'lowest_pub_precision.txt'
            try:
                top1_p, top2_p, top3_p = extract_precision(precision_path)
                precision_acc_top1.append(top1_p)
                precision_acc_top2.append(top2_p)
                precision_acc_top3.append(top3_p)
            except Exception:
                continue

        mean_top1_acc = np.mean(precision_acc_top1, axis=0)
        mean_top2_acc = np.mean(precision_acc_top2, axis=0)
        mean_top3_acc = np.mean(precision_acc_top3, axis=0)

        y1_top1, y2_top1 = compute_fill_between(precision_acc_top1)
        y1_top2, y2_top2 = compute_fill_between(precision_acc_top2)
        y1_top3, y2_top3 = compute_fill_between(precision_acc_top3)
        plt_title = titles[idx]
        plot_accuracy(ax,mean_top1_acc, mean_top1_acc - y1_top1, mean_top1_acc + y2_top1, mean_top2_acc, mean_top2_acc - y1_top2, mean_top2_acc + y2_top2, mean_top3_acc, mean_top3_acc - y1_top3, mean_top3_acc + y2_top3, plt_title, "Training iterations (epochs)", "Mean precision@k", ['Top1', 'Top2', 'Top3'], precision_ylim)
#assemble_lowest_published_precision()
plt.show()

# =================== Plot bar plots for frequency for GRU WC =============================
def plot_freq(y_val, title, xlabel, ylabel, leg):
    x_pos = np.arange(len(y_val))
    plt.plot(x_pos, y_val, color='b')
    plt.title(title, size=size_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #ax.legend(leg, loc=leg_loc, prop={'size': leg_size})
    plt.grid(True)
    plt.show()
  
def assemble_freq(title, file_name, order_tools=None):
    fig = plt.figure(figsize=fig_size)
    tool_freq_dict = dict()
    for i in range(1, runs+1):
        path = base_path + 'gru_wc' + '/run' + str(i) + '/'
        freq_path = path + file_name 
        try:
            with open(freq_path, 'r') as f:
                data = json.loads(f.read())
                for t in data:
                    if t not in tool_freq_dict:
                        tool_freq_dict[t] = list()
                    tool_freq_dict[t].append(data[t])
        except Exception as e:
            print(e)
            continue
    t_names = list()
    t_values = list()
    if not order_tools:
        for t in tool_freq_dict:
            mean_frq = np.mean(tool_freq_dict[t])
            tool_freq_dict[t] = mean_frq
            t_names.append(t)
            t_values.append(mean_frq)
        plot_freq(t_values, title, "Number of tools", "Frequency", [])
    else:
        for t in order_tools:
            mean_frq = np.mean(tool_freq_dict[t])
            t_values.append(mean_frq) 
        plot_freq(t_values, title, "Number of tools", "Frequency", [])
        
order_tools = assemble_freq("Mean frequency (before uniform sampling) of last tools in train tool sequences", 'freq_dict_names.txt')
assemble_freq("Mean frequency (after uniform sampling) of last tools in train tool sequences", 'generated_tool_frequencies.txt', order_tools)

# ================== Plot precision for low freq tools

def plot_scatter(xval, yval1, title, xlabel, ylabel):
    plt.scatter(xval, yval1, c='b')
    #plt.legend(leg, loc=leg_loc, prop={'size': leg_size})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-0.1, 1.1)
    plt.title(title)
    plt.grid(True)
    plt.show()
    

def assemble_low_precision(file_name):
    n_calibrations = 50
    runs = 10
    run_pub_prec = np.zeros((runs, n_calibrations))
    run_norm_prec = np.zeros((runs, n_calibrations))
    run_last_t_freq = np.zeros((runs, n_calibrations))
    run_paths = np.zeros((runs, n_calibrations))
    fig = plt.figure(figsize=fig_size)
    tool_freq_dict = dict()
    for i in range(1, runs+1):
        path = base_path + 'gru_wc' + '/run' + str(i) + '/'
        low_freq_prec_path = path + file_name 
        try:
            with open(low_freq_prec_path, 'r') as f:
                data = f.read()
                split_d = data.split("\t")
                pub_prec = split_d[0]
                norm_prec = split_d[1]
                last_t_mean_freq = split_d[2]
                n_paths = split_d[3]
                
                run_pub_prec[i-1][:] = pub_prec.split(",")
                run_norm_prec[i-1][:] = norm_prec.split(",")
                run_last_t_freq[i-1][:] = last_t_mean_freq.split(",")
                run_paths[i-1][:] = n_paths.split(",")
                
        except Exception as e:
            print(e)
            continue
    mean_pub_prec = np.nanmean(run_pub_prec, axis=0)
    mean_norm_prec = np.nanmean(run_norm_prec, axis=0)
    mean_last_t_freq = np.nanmean(run_last_t_freq, axis=0)
    mean_paths = np.nanmean(run_paths, axis=0)
    
    plt_title = "Mean normal precision@k vs frequencies of last tools"
    plot_scatter(mean_last_t_freq, mean_norm_prec, plt_title, "Frequency of last tools in train tool sequences", "Top 1 precision for test tool sequences")
    plt_title = "Mean standard precision@k vs frequencies of last tools"
    plot_scatter(mean_last_t_freq, mean_pub_prec, plt_title, "Frequency of last tools in train tool sequences", "Top 1 precision for test tool sequences")

assemble_low_precision("test_paths_low_freq_tool_perf.txt")

###########3 Bar plot for extra trees

def read_p(file_p):
    with open(file_p, 'r') as f:
        data = f.read()
        data = data.split("\n")
        data.remove('')
        data = data[0].split(' ')
        data = list(map(float, data))
    return data

def plot_extra_trees():
    normal_path = "data_20_05/extra_trees/precision.txt"
    published_path = "data_20_05/extra_trees/published_precision.txt"

    normal_p = read_p(normal_path)
    published_p = read_p(published_path)
    
    top1_n = normal_p[0]
    top2_n = normal_p[1]
    
    top1_p = published_p[0]
    top2_p = published_p[1]
    
    print(top1_n, top2_n)
    print(top1_p, top2_p)

    fig = plt.figure()
    X = [0.0, 0.2, 0.4, 0.6]
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(0.00, [top1_n], color = 'b', width = 0.1)
    plt.bar(0.2, [top2_n], color = 'b', width = 0.1)
    plt.bar(0.4, [top1_p], color = 'r', width = 0.1)
    plt.bar(0.6, [top2_n], color = 'r', width = 0.1)

    x_ticks = ["Top-1 Normal", "Top-2 Normal", "Top-1 Standard", "Top-2 Standard"]

    plt.ylabel('Precision')
    plt.title('Normal and standard precision@k using ExtraTrees classifier')
    plt.xticks(X)
    plt.xticks(X, x_ticks)
    plt.grid(True)
    plt.show()

plot_extra_trees()

############## Plot data distribution

'''paths_path = 'data/rnn_custom_loss/run1/paths.txt'
all_paths = list()

with open(paths_path) as f:
    all_paths = json.loads(f.read())

path_size = dict()
for path in all_paths:
    path_split = len(path.split(","))
    try:
        path_size[path_split] += 1
    except:
        path_size[path_split] = 1

keys = sorted(list(path_size.keys()))
values = list(path_size.values())

sorted_key_values = list()
sizes = list()
for i, ky in enumerate(keys):
    if i in path_size:
        sizes.append(str(i))
        sorted_key_values.append(path_size[i])
        
def plot_path_size_distribution(x_val, title, xlabel, ylabel, xlabels):
    plt.figure(figsize=fig_size)
    x_pos = np.arange(len(x_val))
    plt.bar(range(len(x_val)), x_val, color='skyblue')
    plt.xlabel(xlabel, size=size_label)
    plt.ylabel(ylabel, size=size_label)
    plt.title(title, size=size_title)
    plt.xticks(x_pos, xlabels, size=size_label)
    plt.yticks(size=size_label)
    plt.grid(True)
    plt.show()

#plot_path_size_distribution(sorted_key_values, 'Data distribution', 'Number of tools in paths', 'Number of paths', sizes)'''

################################################################ Tool usage


'''import csv
import numpy as np
import collections

#import plotly
#import plotly.graph_objs as go
#from plotly import tools
#import plotly.io as pio
from matplotlib import pyplot as plt

def format_tool_id(tool_link):
        tool_id_split = tool_link.split( "/" )
        tool_id = tool_id_split[ -2 ] if len( tool_id_split ) > 1 else tool_link
        return tool_id

tool_usage_file = "../data/tool-popularity-19-09.tsv"
cutoff_date = '2017-12-01'
tool_usage_dict = dict()
tool_list = list()
dates = list()
with open( tool_usage_file, 'rt' ) as usage_file:
    tool_usage = csv.reader(usage_file, delimiter='\t') 
    for index, row in enumerate(tool_usage):
        if (str(row[1]) > cutoff_date) is True:
            tool_id = format_tool_id(row[0])
            tool_list.append(tool_id)
            if row[1] not in dates:
                dates.append(row[1])
            if tool_id not in tool_usage_dict:
                tool_usage_dict[tool_id] = dict()
                tool_usage_dict[tool_id][row[1]] = int(row[2])
            else:
                curr_date = row[1]
                if curr_date in tool_usage_dict[tool_id]:
                    tool_usage_dict[tool_id][curr_date] += int(row[2])
                else:
                    tool_usage_dict[tool_id][curr_date] = int(row[2])
unique_dates = list(set(dates))
for tool in tool_usage_dict:
    usage = tool_usage_dict[tool]
    dts = usage.keys()
    dates_not_present = list(set(unique_dates) ^ set(dts))
    for dt in dates_not_present:
        tool_usage_dict[tool][dt] = 0
    tool_usage_dict[tool] = collections.OrderedDict(sorted(usage.items()))
tool_list = list(set(tool_list))

colors = ['r', 'b', 'g', 'c']
tool_names = ['Cut1', 'cufflinks', 'bowtie2', 'DatamashOps']
legends_tools = ['Tool B', 'Tool C', 'Tool D', 'Tool E']
xticks = ['Jan, 2018', '', 'Mar, 2018', '', 'May, 2018', '', 'Jul, 2018', '', 'Sep, 2018', '', 'Nov, 2018', '', 'Jan, 2019', '', 'Mar, 2019', '', 'May, 2019', '', 'Jul, 2019', '', 'Sep, 2019' ]

def plot_tool_usage(tool_names):
    plt.figure(figsize=(12, 12))
    for index, tool_name in enumerate(tool_names):
        y_val = []
        x_val = []
        tool_data = tool_usage_dict[tool_name]
        for x, y in tool_data.items():
            x_val.append(x)
            y_val.append(y)
        y_reshaped = np.reshape(y_val, (len(x_val), 1))
        plt.plot(y_reshaped[:len(y_reshaped) -1], colors[index])

    plt.legend(legends_tools)
    plt.xlabel('Months', size=size_label)
    plt.ylabel('Usage frequency', size=size_label)
    x_val = x_val[:len(x_val) - 1]
    plt.title("Usage frequency of tools over months")
    plt.xticks(range(len(xticks)), xticks, size=size_label, rotation='30')
    plt.grid(True)
    plt.show()


plot_tool_usage(tool_names)
'''
