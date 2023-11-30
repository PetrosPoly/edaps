import os.path

from PyPDF2 import PdfFileReader, PdfFileWriter, PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Function to merge PDF files into a single PDF file
def merge_pdfs(paths, output):
    # pdf_writer = PdfFileWriter()
    pdf_writer = PdfWriter()

    for path in paths:
        pdf_reader = PdfReader(path)
        # pdf_reader = PdfFileReader(path)

        # for page_num in range(pdf_reader.getNumPages()):
        for page_num in range(len(pdf_reader.pages)):
            # pdf_writer.addPage(pdf_reader.getPage(page_num))
            pdf_writer.add_page(pdf_reader.pages[page_num])

    with open(output, 'wb') as out:
        pdf_writer.write(out)

# Function to combine plots into a single PDF with specified layout
def create_combined_pdf():
    # Create a PDF with reportlab
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)

    # Define positions for the plots
    # You might need to adjust these coordinates depending on your specific requirements
    positions = [(100, 500), (300, 500), (100, 250), (300, 250)]

    src_path = '/media/suman/CVLHDD/apps/experiments/mmgeneration_experiments_on_euler_backup/euler-exp00004/231016_2316_lamp_pix2pix_dnc_3_dbc_32_glr_0.0002_dlr_2e-06_9cc2a/'
    fp1 = os.path.join(src_path, 'loss_total.pdf')
    fp2 = os.path.join(src_path, 'loss_disc.pdf')
    fp3 = os.path.join(src_path, 'loss_pix.pdf')
    fp4 = os.path.join(src_path, 'loss_gen.pdf')
    out_file = os.path.join(src_path, "all_losses.pdf")

    files = [fp1, fp2, fp3, fp4]

    fp1_temp = os.path.join(src_path, 'loss_total_temp.pdf')
    fp2_temp = os.path.join(src_path, 'loss_disc_temp.pdf')
    fp3_temp = os.path.join(src_path, 'loss_pix_temp.pdf')
    fp4_temp = os.path.join(src_path, 'loss_gen_temp.pdf')
    # out_file_temp = os.path.join(src_path, "all_losses.pdf")
    files_temp = [fp1_temp, fp2_temp, fp3_temp, fp4_temp]

    idx = 0
    for file, position in zip(files, positions):
        # First, merge all pages of the current PDF into a single PDF
        # temp_output = f"temp_{file}"
        temp_output = files_temp[idx]
        merge_pdfs([file], temp_output)
        # Draw the temporary merged PDF into the canvas at the specified position
        can.drawImage(temp_output, position[0], position[1], width=200, height=200, mask='auto')
        idx+=1

    can.save()

    # Move the canvas content to a new PDF
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    output_pdf = PdfFileWriter()
    output_pdf.addPage(new_pdf.getPage(0))

    # Save the final output
    with open(out_file, "wb") as f:
        output_pdf.write(f)

# Create the combined PDF
create_combined_pdf()
