from urllib.parse import _NetlocResultMixinStr
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def show_hypothesis(model, data):
    
    if not model:
        print("No Model to plot hypothesis.")
        return None
    preds = [model(x) for x in data.X]
    plt.clf()
    plt.plot(data.X, data.Y, 'o', color='black', label='Raw Data')
    plt.plot(data.X, preds, color='red', label='y=ax+b')
    plt.xlabel('Mileage (x)', color='#1C2833')
    plt.ylabel('Price (y)', color='#1C2833')
    plt.title('Hypothesis')
    plt.legend(loc='upper right')
    plt.show()
    pass

def show_data(model, data):
    normed_x = None
    if model and len(model._modules) > 1:
        normed_x = [model._modules[0](x) for x in data.X]
    plt.clf()
    plt.plot(data.X, data.Y, 'o', color='black', label='Raw Data')
    if normed_x:
        plt.plot(normed_x, data.Y, 'o', color='red', label='Transformed Data')
    plt.xlabel('Mileage (x)', color='#1C2833')
    plt.ylabel('Price (y)', color='#1C2833')
    plt.title('Data Points')
    plt.legend(loc='upper right')
    plt.show()
