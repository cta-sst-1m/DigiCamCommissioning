from tqdm import tqdm
import sys
import logging

class EventCounter:

    def __init__(self, event_min, event_max, log, batch_size=-1, level_dc_min=-1, level_dc_max=-1, level_ac_min=-1, level_ac_max=-1, event_per_level=-1, event_per_level_in_file=-1):

        if level_dc_min == -1 or level_dc_max == -1 or level_ac_min == -1 or level_ac_max == -1 or event_per_level == -1 or event_per_level_in_file == -1:

            level_dc_min = 0
            level_dc_max = 0
            level_ac_min = 0
            level_ac_max = 0
            event_per_level = event_max - event_min
            event_per_level_in_file = event_max - event_min

        self.event_id = -1  # id of event in file
        self.event_id_in_level = -1  # id of event in level
        self.event_count = -1  # count of events outputted
        self.event_count_in_level = -1  # count of events outputted
        self.level_ac = 0  # level id of event
        self.level_dc = 0 # level id of event
        self.event_min = event_min
        self.event_max = event_max
        self.level_ac_min = level_ac_min
        self.level_dc_min = level_dc_min
        self.level_ac_max = level_ac_max
        self.level_dc_max = level_dc_max
        self.event_per_level = event_per_level
        self.event_per_level_in_file = event_per_level_in_file
        self.progress_bar = tqdm(total=event_max)
        self.continuing = False # has to continue loop ?
        self.logger = log

        self.batch_size = batch_size
        self.batch_id = 0
        self.fill_batch = False

        if event_per_level % batch_size != 0:
            raise ValueError(
                'batch_size : %d should be a multiple of event_per_level : %d' % (batch_size, event_per_level))

        if (level_ac_max * level_dc_max * event_per_level_in_file) > event_max:
            raise ValueError('Not enough events in file')

        if batch_size > event_max - event_min:
            raise ValueError('Batch size : %d  > number of events : %d' % (batch_size, event_max - event_min))


    def __iter__(self):
        return self

    def __next__(self):

        self.event_id += 1
        self.event_id_in_level += 1
        self.progress_bar.update(1)
        self.fill_batch = False

        if (self.event_count + 2) % self.batch_size == 0 and not self.event_count == 0:

            self.fill_batch = True
            self.batch_id += 1

        if self.event_id % self.event_per_level_in_file == 0 and self.event_id != 0:

            if self.level_ac >= self.level_ac_max:

                self.level_ac = 0
                self.level_dc += 1

            else:
                self.level_ac += 1
            self.event_id_in_level = 0
            self.logger.debug('Going to AC level %d, DC level %d' % (self.level_ac, self.level_dc))

        if self.event_id < self.event_min:
            self.continuing = True

        elif self.event_id_in_level >= self.event_per_level:
            self.continuing = True

        elif self.level_dc < self.level_dc_min or self.level_ac < self.level_ac_min:

            self.continuing = True

        else:
            self.continuing = False

        if not self.continuing:
            self.event_count += 1
            self.event_count_in_level += 1

        if self.event_id >= self.event_max or self.level_ac > self.level_ac_max or self.level_dc > self.level_dc_max:

            self.logger.error('Stopped iteration')
            raise StopIteration

        else:

            return self

if __name__ == '__main__':
    event_min = 0
    event_max = 360000
    level_dc_min = -1
    level_dc_max = 1
    level_ac_min = 0
    level_ac_max = 1
    event_per_level = 10
    event_per_level_in_file = 1000
    events_per_file = 20000
    batch_size = 10 ## still in dev !
    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    gen = EventCounter(event_min, event_max, log, batch_size, level_dc_min, level_dc_max, level_ac_min, level_ac_max, event_per_level, event_per_level_in_file)

    print(gen.__dict__)

    for file in range(5):

        for counter, event_id_in_file in zip(gen, range(events_per_file)):
            if counter.continuing: continue

            print('file : %d, event_id_in_file : %d, event_id : %d, level_dc : %d, level_ac %d, event_count : %d, event_count_in_level : %d , batch : %s'\
                  % (file, event_id_in_file, counter.event_id, counter.level_dc, counter.level_ac, counter.event_count, counter.event_count_in_level, counter.fill_batch))

