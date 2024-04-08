
if __name__ == "__main__":
    data = pd.read_csv('alfa_dataset.csv')
    data = data.iloc[:,:]
    data = data[['time', 'pitch_measured', 'label', 'airspeed_measured', 'velocity_z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'airspeed', 'groundspeed', 'throttle', 'altitude', 'climb']]
    del data['time']

    for iteraz in range(10):
        np.random.seed(iteraz)
        start = np.random.randint(0, 1600-train_size)
        print(iteraz, "Start", start)
        sc = StandardScaler()
        pca = PCA(n_components = 3)

        data_train = data[start:start+train_size]
        del data_train['label']
        np.random.seed(0)
        data_train = sc.fit_transform(data_train)
        np.random.seed(0)
        data_train = pca.fit_transform(data_train)

        data_test = data[1600:]
        gt = np.array(data_test['label'])
        gt = gt[w-1:]
        del data_test['label']
        data_test = sc.transform(data_test)
        data_test = pca.transform(data_test)
