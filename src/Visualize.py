import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def show_hypothesis(model, data):

    pass

def show_data(model, data):
    if len(model._modules) > 1:
        normed_x = [model._modules[0](x) for x in data.X]
        print(normed_x)
        input()
    plt.clf()
    plt.plot(data.X, data.Y, 'o', color='black')
    plt.plot(normed_x, data.Y, 'o', color='red')
    plt.show()

def show_derivative(model, x, y):
    
    pass

