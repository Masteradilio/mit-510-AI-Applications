from markdown_pdf import MarkdownPdf, Section

input_markdown_path = 'c:\\Users\\Adilio\\Documents\\Projetos\\MIT-510\\reports\\relatorio_projeto.md'
output_pdf_path = 'c:\\Users\\Adilio\\Documents\\Projetos\\MIT-510\\docs\\relatorio_projeto.pdf'

with open(input_markdown_path, 'r', encoding='utf-8') as f:
    markdown_content = f.read()

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(markdown_content, toc=False))
pdf.save(output_pdf_path)

print(f"Arquivo PDF salvo em: {output_pdf_path}")