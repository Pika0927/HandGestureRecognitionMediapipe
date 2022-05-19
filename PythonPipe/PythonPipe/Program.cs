using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SWF = System.Windows.Forms;
namespace PythonPipe
{
    internal class Program
    {
        [System.Runtime.InteropServices.DllImport("user32")]
        private static extern int mouse_event(int dwFlags, int dx, int dy, int cButtons, int dwExtraInfo);
        const int MOUSEEVENTF_MOVE = 0x0001;
        const int MOUSEEVENTF_LEFTDOWN = 0x0002;
        const int MOUSEEVENTF_LEFTUP = 0x0004;
        const int MOUSEEVENTF_RIGHTDOWN = 0x0008;
        const int MOUSEEVENTF_RIGHTUP = 0x0010;
        const int MOUSEEVENTF_MIDDLEDOWN = 0x0020;
        const int MOUSEEVENTF_MIDDLEUP = 0x0040;
        const int MOUSEEVENTF_ABSOLUTE = 0x8000;

        static void Main(string[] args)
        {
            Program Self = new Program();
            Self.run_server();
            Console.ReadKey(true);
        }
        void run_server()
        {
            // Open the named pipe.
            var server = new NamedPipeServerStream("NPGesture");

            Console.WriteLine("Waiting for connection...");
            server.WaitForConnection();

            Console.WriteLine("Connected.");
            var br = new BinaryReader(server);
            //var bw = new BinaryWriter(server);
            int SameCounter = 0;
            double PreX = -1;
            double PreY = -1;
            double PosX = -1;
            double PosY = -1;
            string PreID = "";

            double WindowH = 1080;
            while (true)
            {
                try
                {
                    var len = (int)br.ReadUInt32();            // Read string length
                    var str = new string(br.ReadChars(len));    // Read string
                    Console.WriteLine($"Read:{str} Time:{DateTime.Now} {DateTime.Now.Millisecond}");
                    if (str[0] != 'x')
                    {
                        string[] RowData = str.Split(',');
                        if (RowData[0] == PreID)
                        {

                            SameCounter += SameCounter > 10 ? 0 : 1;

                            if (SameCounter > 3)
                            {
                                if (RowData[0] == "1")
                                {
                                    if (PreX < 0)
                                    {
                                        PreX = Convert.ToDouble(RowData[1]);
                                        PreY = Convert.ToDouble(RowData[2]);
                                        continue;
                                    }
                                    PosX = Convert.ToDouble(RowData[1]);
                                    PosY = Convert.ToDouble(RowData[2]);
                                    double PLen = GetLength(
                                        PosX,
                                        PosY,
                                        Convert.ToDouble(RowData[3]),
                                        Convert.ToDouble(RowData[4])
                                        );
                                    double Rate = WindowH / PLen;
                                    SWF.Cursor.Position = new System.Drawing.Point(
                                        SWF.Cursor.Position.X + (int)((PosX - PreX) * Rate),
                                        SWF.Cursor.Position.Y + (int)((PosY - PreY) * Rate));
                                    PreX = Convert.ToDouble(RowData[1]);
                                    PreY = Convert.ToDouble(RowData[2]);
                                }
                                else
                                {
                                    if (RowData[0] == "5" && SameCounter == 5)
                                    {
                                        mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                                    }
                                    PreX = -1;
                                    PreY = -1;
                                }
                            }

                        }
                        else
                        {
                            SameCounter = 0;
                            PreID = RowData[0];
                        }
                    }
                    else
                    {
                        PreID = "";
                        PreX = -1;
                        PreY = -1;
                    }

                }
                catch (EndOfStreamException)
                {
                    break;                    // When client disconnects
                }
            }

            Console.WriteLine("Client disconnected.");
            server.Close();
            server.Dispose();
        }
        double GetLength(double x1, double y1, double x2, double y2)
        {
            return Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }
    }
}
