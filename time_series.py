from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt


class TimeSeries:
    """Encapsulates file parsing, computation, and plotting for library data.

    Creates a time series analysis for the Monthly Wifi Sessions at Chicago 
    Public Libraries data set. Includes file parsing for reading the CSV into
    Numpy arrays, computation for creating a moving average and least squares
    regression, and plots the data too. 

    Args: 
        file : Takes in the Chicago library data file to parse.

    Attributes:
        num_sessions : A numpy array that contains the number of sessions.
        date : A numpy array that contains both the year and month for each 
            session. To be used for date_list.
        date_list : A list that combines the month and year for future plotting.

    """

    def __init__(self, file: str) -> None:
        self.file: str = file
        self.num_sessions: np.ndarray = np.loadtxt(file,
                                                   delimiter=',',
                                                   usecols=2,
                                                   skiprows=1)
        self.date: np.ndarray = np.loadtxt(file,
                                           delimiter=",",
                                           dtype='str',
                                           skiprows=1,
                                           usecols=(0, 1))

        self.date_list: List = [' '.join(i) for i in self.date]

    def plot_series(self) -> None:
        """Plots the date and number of wifi sessions. 
        """
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.title("Number of Monthly WiFi Sessions at CPL")
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("Number of Sessions", fontsize=15)
        plt.plot(self.date_list, self.num_sessions)
        plt.xticks(rotation=90, horizontalalignment='right')
        plt.tight_layout()

    def moving_avg(self, m: int) -> np.ndarray:
        """Creates a moving average array based on an order value, m.

        Args: 
            m : The order value for the moving average. 

        Returns:
            A numpy array of the completed moving average array.

        """
        moving_avg_array = []
        for i in range(len(self.num_sessions)-m+1):
            ma_n = 1/m * (np.sum(self.num_sessions[i:m+i]))
            moving_avg_array.append(ma_n)
        return np.array(moving_avg_array)

    def plot_ma(self, ma: np.ndarray) -> None:
        """Plots the most recently created Moving Average estimate.

        Args: 
            ma : A moving average array, created from the moving_avg() function.

        """
        k = (len(self.num_sessions) - len(ma)) // 2
        fig = plt.figure(figsize=(6, 3), dpi=100)
        plt.title("Moving Average vs Number of Monthly WiFi Sessions at CPL")
        plt.xlabel("TimeStep (N)")
        plt.ylabel("Number of Sessions")
        plt.plot(np.arange(k, k + len(ma)),
                 self.num_sessions[k: - k], label='Monthly Sessions')
        plt.plot(np.arange(k, k + len(ma)),
                 ma, label='Moving Average')
        plt.legend()

    def lstsqr(self, mov_avg: np.ndarray, intercept: bool = True) -> Dict:
        """Fits the moving average to a linear model using least squares.

        Uses the numpy linalg.lstsq function to create the least squares 
        regression from the moving average model. Can be made with or without
        an intercept.

        Args:
            mov_avg : A moving average array created from moving_avg().
            intercept : decides if an intercept B0 is included or not. 
                Default is set to True.

        Returns:
            A dictionary with the keys equal to the betas "B0" or "B1" and the 
                value corresponding to that beta value. 

        """
        k = (len(self.num_sessions) - len(mov_avg)) // 2
        time_step = np.arange(k, len(mov_avg) + k)

        if intercept:
            A = np.append(np.ones((len(mov_avg), 1)),
                          time_step.reshape(len(mov_avg), 1), axis=1)
            betas, Resid, _, ew_ = np.linalg.lstsq(A, mov_avg, rcond=None)
            print(f"β0 = {betas[0]}")
            print(f"β1 = {betas[1]}")
            return {'B0': betas[0], 'B1': betas[1]}
        else:
            beta1, Resid, _, _ = np.linalg.lstsq(
                time_step.reshape(len(mov_avg), 1), mov_avg, rcond=None)
            print(f"β1 = {beta1[0]}")
            return {'B1': beta1[0]}

    def lowest_resid(self, model1: Dict, model2: Dict) -> None:
        """Calculates the best model based on the lower SSR value.

        Prints the Sum of Squared residual value for model 1 and model 2.
        The user can then assess which model is better.

        Args: 
            model1 : First model input, created from the lstsqr() function.
            model2 : Second model input, also created from lstsqr() function.

        """
        y_true = self.num_sessions
        y_hat1 = np.arange(len(self.num_sessions)) * model1['B1']
        y_hat2 = model2['B0'] + \
            (np.arange(len(self.num_sessions)) * model2['B1'])

        m1SSR = np.sum((y_true - y_hat1)**2)
        m2SSR = np.sum((y_true - y_hat2)**2)

        print(f'model 1 SSR {m1SSR}')
        print(f'model 2 SSR {m2SSR}')

    def model_compar(self, model1: Dict, model2: Dict) -> None:
        """Plots the regression estimate from both model 1 and model 2.
        """
        fig, ([ax1, ax2]) = plt.subplots(1, 2, figsize=(10, 6))
        ax1.plot(self.num_sessions)
        ax1.plot(np.arange(len(self.num_sessions)) * model1['B1'])
        ax1.set_title('Model 1')
        ax1.set_ylabel('Number of Sessions')
        ax1.set_xlabel('Time Step (N)')
        ax2.plot(self.num_sessions)
        ax2.plot(model2['B0'] +
                 (np.arange(len(self.num_sessions)) * model2['B1']))
        ax2.set_title('Model 2')
        plt.legend(['Number of Sessions', 'Least Squares Estimate'])
        plt.xlabel('Time Step (N)')
