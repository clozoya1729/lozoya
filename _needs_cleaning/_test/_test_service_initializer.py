from _misc.service_initializer import SERVICES, initialize_service

# SERVER = r'/mnt' #for server
SERVER = r'Z:/Family/LoParTechnologies/PythonServer'  # for local


def test_audio_production():
    kwargs = {
        'filesDir': f'{SERVER}\AudioTest', 'service': SERVICES[0],
        'server':   SERVER, 'jobName': 'Test Artist - Test Song',
    }

    initialize_service(**kwargs)


def test_report_generator():
    kwargs = {
        'filesDir':   r'{}/data/ReportTest'.format(SERVER),
        'service':    SERVICES[1],
        'server':     SERVER,
        'jobName':    'Test report_generator0',
        'formatting': 'APA',
        'pairPlots':  True,
    }
    initialize_service(**kwargs)
