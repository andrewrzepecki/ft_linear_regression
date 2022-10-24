from urllib.parse import _NetlocResultMixinStr
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def show_hypothesis(model, data):
    colors = ['black', 'blue']
    colors_t = ['red', 'green']
    if not model:
        print("No Model to plot hypothesis.")
        return None
    preds = [model(x) for x in data.X]
    plt.clf()
    fig, axs = plt.subplots(model.input_size)
    fig.suptitle("Hypothesis")
    for i in range(model.input_size):
        a = axs[i] if model.input_size > 1 else axs 
        a.plot([d[i] for d in data.X], data.Y, 'o', color=colors[i], label=f'Raw Data [{i}]')
        a.plot([d[i] for d in data.X], preds, color=colors_t[i], label=f'y=ax+b [{i}]')
        a.legend(loc='upper right')
        fig.show()

def show_data(model, data):
    colors = ['black', 'blue']
    colors_t = ['red', 'green']
    normed_x = None
    if model and len(model._modules) > 1:
        normed_x = [model._modules[0](x) for x in data.X]
    plt.clf()
    for i in range(model.input_size):
        plt.plot([d[i] for d in data.X], data.Y, 'o', color=colors[i], label=f'Raw Input [{i}]')
    if normed_x:
        for i in range(model.input_size):
            plt.plot([d[i] for d in normed_x], data.Y, 'o', color=colors_t[i], label=f'Transformed Data [{i}]')
    plt.xlabel('Mileage (x)', color='#1C2833')
    plt.ylabel('Price (y)', color='#1C2833')
    plt.title('Data Points')
    plt.legend(loc='upper right')
    plt.show()
