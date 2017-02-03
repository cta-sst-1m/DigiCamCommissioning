import numpy as np
import h5py


class telescope:
    def __init__(self, id,):
        self.eventNumber = -1
        self.adc_samples = {}
        self.id = id
class r1:
    def __init__(self,id_list = [0], event_id=0):
        self.tels_with_data = []
        self.event_id = event_id
        self.tel = {}
        for id in id_list:
            self.tels_with_data.append(id)
            self.tel[id]=telescope(id)

    def set_data(self,telid,evtnum,adcs):
        self.tel[telid].eventNumber = evtnum
        self.tel[telid].adc_samples = {i: adcs[i] for i in range(len(adcs))}

class ToyReader: # create a reader as asked
    def __init__(self, filename='../../digicamtoy/data_calibration_cts/toy_data_', id_list = [0], max_events=50000, n_pixel=1296, weights=1., seed=0):
        self.count = 0
        self.event_id = 0
        self.r1 = r1(id_list, event_id=self.count)
        self.evt_max = max_events
        self.id_list = id_list
        #open file
        self.filename = filename
        self.n_pixel = n_pixel
        self.hdf5_file = h5py.File(self.filename, 'r')
        self.weights = weights
        self.seed = seed
        np.random.seed(seed=self.seed)


        return


    def __iter__(self):
        return self

    def __next__(self):
         return self.next()


    def next(self):
        #aller a ton evt suivant dan le fichier: count+=1
        ### convertir adcs en list(list()) (pixel*sample)


        if self.count<self.weights*self.evt_max:

            #self.hdf5_file = h5py.File(self.filename + str(self.count) + '.hdf5', 'r')
            n_traces_tot = self.hdf5_file['data']['traces'].shape[0]
            n_samples = self.hdf5_file['data']['traces'].shape[1]

            adcs = []

            i = 0

            while i<self.n_pixel:

                random_samples = np.random.randint(low=0, high=n_traces_tot, size=1)
                adcs.append(self.hdf5_file['data']['traces'][random_samples[0]])
                i = i + 1

            #print('n_signal_photon', self.hdf5_file['param']['n_signal_photon'].value, end='\r')

            adcs = np.array(adcs).reshape(self.n_pixel, n_samples)

            for telid in self.id_list:
                self.r1.set_data(telid, self.event_id, adcs)

            self.count += 1

        else:

            self.hdf5_file.close()
            raise StopIteration

        return self


if __name__ == '__main__':

    _url = '../../digicamtoy/data_calibration_cts/toy_data_0.hdf5'
    inputfile_reader = ToyReader(filename=_url, id_list=[0], max_events=2, weights=1.)
    i = 0

    print('event.r1.tels_with_data', inputfile_reader.r1.tels_with_data)

    for event in inputfile_reader:

        print(i)
        print('event.r1.tels_with_data',event.r1.tels_with_data)
        telid=0
        print('event.r1.tel[telid].eventNumber',event.r1.tel[telid].eventNumber)
        print('adcs',np.array(list(event.r1.tel[telid].adc_samples.values())))
        print('adcs max in pixel 0',np.max(np.array(list(event.r1.tel[telid].adc_samples.values()))[0,:]))
        print ('adcs shape : ',np.array(list(event.r1.tel[telid].adc_samples.values())).shape)


        i = i+1

    print (i)