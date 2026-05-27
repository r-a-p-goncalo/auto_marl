

import os
from automarl.components.loggers.logger_component import DEBUG_LEVEL, LoggerSchema


class NotInComponentTree(Exception):
    '''Component not found in tree of another component'''






def common_exception_handling(logger : LoggerSchema, exception, error_report_path):

    '''
    Common exception handling for components with logging
    It is about saving the error in file
    '''

    import traceback
        
    error_message = str(exception)
    full_traceback = traceback.format_exc()

    logger.writeLine("Error message:", file=error_report_path, level=DEBUG_LEVEL.ERROR)
    logger.writeLine(error_message, file=error_report_path, level=DEBUG_LEVEL.ERROR)

    logger.writeLine("\nFull traceback:", file=error_report_path, level=DEBUG_LEVEL.ERROR)
    logger.writeLine(full_traceback, file=error_report_path, level=DEBUG_LEVEL.ERROR)