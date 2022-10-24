import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-whitegrid')


def show_3d_data(model, data):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = data.X.T[0]
    ys = data.Y
    zs = data.X.T[1]
    ax.scatter(xs, ys, zs, marker='o', color='black', label='Raw Data')
    if model and len(model._modules) > 1:
        normed_x = np.array([model._modules[0](x) for x in data.X])
        ax.scatter(normed_x.T[0], ys, normed_x.T[1], marker='o', color='green', label='Transformed Data')
    ax.set_xlabel('Mileage (X)')
    ax.set_ylabel('Price (Y)')
    ax.set_zlabel('Year (Z)')
    fig.legend(loc='upper right')
    fig.show()

def show_3d_hypothesis(model, data):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = data.X.T[0]
    ys = data.Y
    zs = data.X.T[1]
    yp = [model(y) for y in data.X]
    ax.scatter(xs, ys, zs, marker='o', color='black', label='Raw Data Used in Training')
    ax.scatter(xs, yp, zs, marker='x', label='Predictions', color='red')

    ax.set_xlabel('Mileage (X)')
    ax.set_ylabel('Price (Y)')
    ax.set_zlabel('Year (Z)')
    fig.legend(loc='upper right')
    plt.title(f'Hypothesis | Loss: {model.get_loss(data.X, data.Y)}')
    fig.show()


def show_hypothesis(model, data):
    
    plt.clf()
    if not model:
        print("No Model to plot hypothesis.")
        return None
    if data.input_size > 1:
        show_3d_hypothesis(model, data)
        return
    preds = [model(x) for x in data.X]
    plt.plot(data.X, data.Y, 'o', color='black', label=f'Raw Data Used in Training')
    plt.plot(data.X, preds, color='red', label=f'y=ax+b')
    plt.xlabel('Mileage (X)', color='#1C2833')
    plt.ylabel('Price (Y)', color='#1C2833')
    plt.title(f'Hypothesis | Loss: {model.get_loss(data.X, data.Y)}')
    plt.legend(loc='upper right')
    plt.show()


def show_data(model, data):
    
    plt.clf()
    if data.input_size > 1:
        show_3d_data(model, data)
        return
    normed_x = None
    if model and len(model._modules) > 1:
        normed_x = [model._modules[0](x) for x in data.X]
    plt.plot(data.X, data.Y, 'o', color='black', label=f'Raw Data')
    if normed_x:
        plt.plot(normed_x, data.Y, 'o', color='green', label=f'Transformed Data')
    plt.xlabel('Mileage (X)', color='#1C2833')
    plt.ylabel('Price (Y)', color='#1C2833')
    plt.title('Data Points')
    plt.legend(loc='upper right')
    plt.show()
