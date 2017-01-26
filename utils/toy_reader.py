import numpy

class telescope:
    def __init__(self, id,):
        self.eventNumber = -1
        self.adc_samples = {}
        self.id = id
class r1:
    def __init__(self,id_list = [0]):
        self.tels_with_data = []
        self.tel = {}
        for id in id_list:
            self.tels_with_data.append(id)
            self.tel[id]=telescope(id)

    def set_data(self,telid,evtnum,adcs):
        self.tel[telid].eventNumber = evtnum
        self.tel[telid].adc_samples = {i: adcs[i] for i in range(len(adcs))}

class ToyReader:
    def __init__(self,filename,id_list = [0]):
        self.count = 0
        self.r1 = r1(id_list)
        self.id_list = id_list
        #open file
        return


    def __iter__(self):
        return self

    def __next__(self):
         return self.next()


    def next(self):
        #aller a ton evt suivant dan le fichier: count+=1
        self.count += 1
        ### convertir adcs en list(list()) (pixel*sample)
        adcs = None
        for telid in self.id_list:
            self.r1.set_data(telid,self.count,adcs)


if __name__ == '__main__':


    inputfile_reader = ToyReader(filename=_url, id_list=[0])
    for event in inputfile_reader:
        print('event.r1.tels_with_data',event.r1.tels_with_data)
        telid=0
        print('event.r1.tel[telid].eventNumber',event.r1.tel[telid].eventNumber)
        print('adcs',np.array(list(event.r1.tel[telid].adc_samples.values())))