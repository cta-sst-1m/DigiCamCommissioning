from tqdm import tqdm


class EventCounter:

    def __init__(self, event_min, event_max, level_min, level_max, event_per_level, event_per_level_in_file, log):

        self.event_id = -1  # id of event in file
        self.event_id_in_level = -1  # id of event in level
        self.event_count = -1  # count of events outputted
        self.event_count_in_level = -1  # count of events outputted
        self.level = 0  # level id of event
        self.event_min = event_min
        self.event_max = event_max
        self.level_min = level_min
        self.level_max = level_max
        self.event_per_level = event_per_level
        self.event_per_level_in_file = event_per_level_in_file
        self.progress_bar = tqdm(total=self.event_max)
        self.continuing = False # has to continue loop ?
        self.logger = log

    def __iter__(self):
        return self

    def __next__(self):

        self.event_id += 1
        self.event_id_in_level += 1
        self.progress_bar.update(1)

        if self.event_id % self.event_per_level_in_file == 0 and self.event_id != 0:
            self.logger.debug('Level %d passed' % self.level)
            self.level += 1
            self.event_id_in_level = 0
            self.event_count_in_level = -1

        if self.event_id < self.event_min:
            self.continuing = True

        elif self.event_id_in_level >= self.event_per_level:
            self.continuing = True

        elif self.level < self.level_min:
            self.continuing = True

        else:
            self.continuing = False

        if not self.continuing:
            self.event_count += 1
            self.event_count_in_level += 1

        if self.event_id >= self.event_max or self.level > self.level_max:
            self.logger.error('Stopped iteration event_id %d >= %d event_max' % (self.event_id, self.event_max))
            raise StopIteration

        else:

            return self

if __name__ == '__main__':
    event_min = 0
    event_max = 60000
    level_min = 0
    level_max = 5
    event_per_level = 10
    event_per_level_in_file = 10000
    events_per_file = 17000

    gen = EventCounter(event_min, event_max, level_min, level_max, event_per_level, event_per_level_in_file)

    for file in range(5):

        for counter, event_id_in_file in zip(gen, range(events_per_file)):
            if counter.continuing: continue

            print('file : %d, event_id_in_file : %d, event_id : %d, level : %d, event_count : %d, event_count_in_level : %d '\
                  % (file, event_id_in_file, counter.event_id, counter.level, counter.event_count, counter.event_count_in_level))
