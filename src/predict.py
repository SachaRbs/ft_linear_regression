import pandas as pd

def predict_(t0, t1, mileage):
    pred = t0 + (t1 * mileage)
    return pred
    # return None
	
def main():
    theta = pd.read_csv(r'../data/theta.csv')
    t0 = float(theta.columns[0])
    t1 = float(theta.columns[1])
    print("enter the mileage of the house...")
    try:
        mileage = int(input())
    except ValueError:
        print("please enter a number")
    pred = predict_(t0, t1, mileage)
    print("prediction : {}".format(pred))


if __name__ == "__main__":
    main()