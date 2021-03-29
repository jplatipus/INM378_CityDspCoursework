
# dummy control signal generator: always returns the same
class MockControlClass:
    def getControlFactor(self, controlIndex):
        return 1

    # interpolate 2 filters using the given controlFactor (between 0 and 1)
    # return a the interpolated filter
    def interpolateFilters(self, controlIndex, filterMin, filterMax):
        return filterMin