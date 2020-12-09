

class GDriveproxy():
  def __init__(self, mount):
    self.mount = mount

  def readFile(self, filename):
    filepath = f"{self.mount}/filename"
